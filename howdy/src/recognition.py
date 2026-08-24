# Shared face recognition backend for Howdy
#
# Wraps InsightFace (SCRFD detector + ArcFace recognizer) running on onnxruntime,
# producing 512-d L2-normalized embeddings matched by cosine similarity.
# This replaces the old dlib pipeline (HOG/CNN detector + 5-point shape predictor
# + 128-d ResNet descriptor matched by euclidean distance).

import os
import io
import json
import contextlib
import warnings

import numpy as np

# insightface aligns every face through a scikit-image call deprecated in 0.26,
# which would warn into the password prompt on each authentication. Scoped to that
# one message so other deprecations still surface.
warnings.filterwarnings("ignore", message="`estimate` is deprecated", category=FutureWarning)

import paths_factory
from i18n import _

# Version stamp written into every model file. Bumped from the implicit "dlib 128-d"
# format so that legacy models are detected and rejected instead of silently misfiring.
MODEL_VERSION = 2
# Dimensionality of an ArcFace embedding
EMBED_DIM = 512

# Preference order used when execution_provider is set to "auto"
_AUTO_PROVIDER_PREFERENCE = [
	"CUDAExecutionProvider",
	"MIGraphXExecutionProvider",
	"ROCMExecutionProvider",
	"CPUExecutionProvider",
]

# Named providers the user can request explicitly in the config.
# For AMD GPUs, MIGraphX is the current execution provider; the older ROCm EP
# was removed from onnxruntime after 1.22. "rocm" tries both for compatibility.
_NAMED_PROVIDERS = {
	"cpu": ["CPUExecutionProvider"],
	"cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
	"migraphx": ["MIGraphXExecutionProvider", "CPUExecutionProvider"],
	"rocm": ["MIGraphXExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider"],
}


class LegacyModelError(Exception):
	"""Raised when a model file predates the ONNX/512-d format and must be re-enrolled."""


def resolve_providers(config):
	"""Determine the onnxruntime execution providers to use.

	Returns a list of provider names, always ending in CPUExecutionProvider as a
	fallback so a stock install keeps working even when a GPU provider was requested
	but is not available in the installed onnxruntime build.
	"""
	import onnxruntime as ort

	available = ort.get_available_providers()
	requested = config.get("core", "execution_provider", fallback="auto").strip().lower()

	if requested == "auto":
		preference = _AUTO_PROVIDER_PREFERENCE
	elif requested in _NAMED_PROVIDERS:
		preference = _NAMED_PROVIDERS[requested]
	else:
		print(_("Unknown execution_provider '{p}', falling back to auto").format(p=requested))
		preference = _AUTO_PROVIDER_PREFERENCE

	# Keep only providers actually compiled into this onnxruntime build
	providers = [p for p in preference if p in available]

	# Guarantee a CPU fallback is always present
	if "CPUExecutionProvider" not in providers:
		providers.append("CPUExecutionProvider")

	return providers


class _Analyzer:
	"""Minimal replacement for insightface's FaceAnalysis.

	FaceAnalysis builds an inference session for every model in the pack only
	to throw most of them away; this holds just the detection and recognition
	models and mirrors the FaceAnalysis get()/prepare() behavior for them.
	"""

	def __init__(self, models):
		self.models = models
		self.det_model = models["detection"]

	def prepare(self, ctx_id, det_thresh, det_size):
		for taskname, model in self.models.items():
			if taskname == "detection":
				model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
			else:
				model.prepare(ctx_id)

	def get(self, img, max_num=0):
		from insightface.app.common import Face

		bboxes, kpss = self.det_model.detect(img, max_num=max_num, metric="default")
		faces = []
		for i in range(bboxes.shape[0]):
			face = Face(
				bbox=bboxes[i, 0:4],
				kps=kpss[i] if kpss is not None else None,
				det_score=bboxes[i, 4],
			)
			self.models["recognition"].get(img, face)
			faces.append(face)
		return faces


def _load_models(pack_dir, providers, provider_options):
	"""Build the detection and recognition model sessions for a pack.

	Uses a task-to-file map stored next to the pack to only construct the two
	sessions that are needed. Without the map (first ever run), every onnx file
	in the pack is routed like FaceAnalysis does and the map is written for the
	next time.
	"""
	import glob
	from insightface import model_zoo

	kwargs = {"providers": providers}
	if provider_options is not None:
		kwargs["provider_options"] = provider_options

	# model_zoo.get_model prints the provider list for every session it builds,
	# which would end up in the middle of the sudo/login prompt
	def get_model(path):
		with contextlib.redirect_stdout(io.StringIO()):
			return model_zoo.get_model(path, **kwargs)

	map_path = os.path.join(pack_dir, "howdy_model_map.json")
	models = {}

	try:
		with open(map_path) as mapfile:
			model_map = json.load(mapfile)
		for task in ("detection", "recognition"):
			models[task] = get_model(os.path.join(pack_dir, model_map[task]))
	except (OSError, KeyError, ValueError):
		models = {}

	if not models:
		# No usable map: route every model in the pack and keep what we need
		for onnx_file in sorted(glob.glob(os.path.join(pack_dir, "*.onnx"))):
			model = get_model(onnx_file)
			if model is None:
				continue
			if model.taskname in ("detection", "recognition") and model.taskname not in models:
				models[model.taskname] = model
			else:
				del model

		# Remember the mapping for fast startup next time (best effort, the
		# pack dir may not be writable for unprivileged commands)
		try:
			with open(map_path, "w") as mapfile:
				json.dump({task: os.path.basename(model.model_file) for task, model in models.items()}, mapfile)
		except OSError:
			pass

	return models


def create_analyzer(config):
	"""Build and prepare the face analyzer from config values.

	Loads only the detection and recognition sub-models (gender/age and dense
	landmark models are not needed) and never downloads at runtime: if the model
	pack is missing it raises with instructions to run the installer.
	"""
	pack = config.get("core", "model_pack", fallback="buffalo_s")

	# Refuse to silently hit the network during authentication
	pack_dir = paths_factory.model_pack_dir_path(pack)
	if not os.path.isdir(pack_dir):
		print(_("Face model pack '{pack}' has not been downloaded, please run the following commands:").format(pack=pack))
		print("\n\tcd " + paths_factory.models_data_dir_path())
		print("\tsudo ./install.sh\n")
		raise FileNotFoundError(pack_dir)

	providers = resolve_providers(config)
	provider_options = None

	if "MIGraphXExecutionProvider" in providers:
		# Work around missing stream synchronization in the MIGraphX provider
		# on APUs (observed on gfx1150 with onnxruntime_migraphx 1.23.2):
		# without this, every inference returns the PREVIOUS inference's output
		# buffer. Blocking kernel launches force completion before results are
		# read. Must be set before the HIP runtime initializes, so before the
		# first session is built.
		os.environ.setdefault("HIP_LAUNCH_BLOCKING", "1")

		# The model cache makes MIGraphX compile each model on the first ever
		# run (slow, see the howdy compile command) and load the compiled
		# binary afterwards, keyed by model and input shape
		cache_dir = paths_factory.compiled_models_dir_path(pack)
		if not os.path.isdir(cache_dir) or not os.listdir(cache_dir):
			print(_("Compiling face models for the GPU, this one-time step can take several minutes..."))
		os.makedirs(cache_dir, exist_ok=True)

		provider_options = [
			{"migraphx_model_cache_dir": cache_dir} if p == "MIGraphXExecutionProvider" else {}
			for p in providers
		]

	models = _load_models(pack_dir, providers, provider_options)
	if "detection" not in models or "recognition" not in models:
		print(_("Face model pack '{pack}' is missing a detection or recognition model").format(pack=pack))
		raise FileNotFoundError(pack_dir)

	analyzer = _Analyzer(models)

	det_size = config.getint("core", "det_size", fallback=640)
	det_thresh = config.getfloat("core", "det_thresh", fallback=0.5)

	# ctx_id 0 selects the first GPU, -1 forces CPU
	ctx_id = 0 if providers[0] != "CPUExecutionProvider" else -1

	analyzer.prepare(ctx_id=ctx_id, det_thresh=det_thresh, det_size=(det_size, det_size))

	# Force compilation/cache-load and GPU buffer allocation now instead of
	# mid-detection, and flush the first inferences, which can return garbage
	# right after a MIGraphX compile
	if providers[0] == "MIGraphXExecutionProvider":
		_warmup(analyzer, det_size)

	# Remember what actually got attached for diagnostics: sessions fall back
	# to CPU silently when a GPU provider fails to load, so ask a live session
	# instead of trusting the requested list
	analyzer.howdy_providers = analyzer.models["detection"].session.get_providers()
	return analyzer


def _warmup(analyzer, det_size):
	"""Run dummy inferences through every model session.

	Triggers the lazy MIGraphX compile (or cache load) at startup and discards
	the first outputs, which can be garbage directly after compilation. Uses
	random data and two passes: all-zero inputs do not exercise the full
	numerical path and have been seen to leave the first real inference wrong.
	"""
	shapes = {
		"detection": (1, 3, det_size, det_size),
		"recognition": (1, 3, 112, 112),
	}
	rng = np.random.default_rng(0)
	for task, model in analyzer.models.items():
		shape = shapes.get(task)
		if shape is None:
			continue
		input_name = model.session.get_inputs()[0].name
		for _pass in range(2):
			dummy = rng.standard_normal(shape).astype(np.float32)
			model.session.run(None, {input_name: dummy})


def get_faces(analyzer, frame):
	"""Return detected faces for a BGR frame.

	Each face exposes .normed_embedding (512-d, L2-normalized), .kps (5 landmark
	points), .bbox and .det_score.
	"""
	return analyzer.get(frame)


def best_match(embedding, encodings):
	"""Find the closest enrolled encoding by cosine similarity.

	Both the query embedding and the stored encodings are L2-normalized, so the dot
	product equals cosine similarity. Returns (index, score) with score in [-1, 1]
	where higher is a better match.
	"""
	similarities = encodings @ embedding
	match_index = int(np.argmax(similarities))
	return match_index, float(similarities[match_index])


def load_encodings(path):
	"""Load and validate a user's model file.

	Returns the parsed list of model records. Raises FileNotFoundError if the file
	does not exist and LegacyModelError if it uses the old dlib 128-d format.
	"""
	with open(path) as datafile:
		models = json.load(datafile)

	for model in models:
		# Old dlib files have no version key and carry 128-d vectors
		if model.get("version") != MODEL_VERSION:
			raise LegacyModelError(path)
		for vector in model["data"]:
			if len(vector) != EMBED_DIM:
				raise LegacyModelError(path)

	return models


def flatten_encodings(models):
	"""Turn a list of model records into a single normalized (N, 512) array.

	Also returns a parallel list mapping each row back to its model record so the
	winning row can be traced to a labelled model.
	"""
	vectors = []
	owners = []
	for model in models:
		for vector in model["data"]:
			vectors.append(vector)
			owners.append(model)
	return np.array(vectors, dtype=np.float32), owners


def new_model_record(label, model_id, embeddings):
	"""Build a version-stamped model record for storage."""
	import time

	return {
		"version": MODEL_VERSION,
		"time": int(time.time()),
		"label": label,
		"id": model_id,
		"data": [np.asarray(e, dtype=np.float32).tolist() for e in embeddings],
	}
