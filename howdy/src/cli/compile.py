# Precompile the face models for the configured execution provider

# Import required modules
import configparser
import sys
import time

import paths_factory
import recognition
from i18n import _

# Read config from disk
config = configparser.ConfigParser()
config.read(paths_factory.config_file_path())

providers = recognition.resolve_providers(config)

# Compiled-model caching only applies to the MIGraphX (AMD GPU) provider
if providers[0] != "MIGraphXExecutionProvider":
	print(_("The configured execution provider ({provider}) does not need a compile step").format(provider=providers[0]))
	sys.exit(0)

print(_("Compiling face models for the GPU, this can take several minutes per model"))
print(_("Already compiled models are loaded from cache and take no time\n"))

start = time.time()

# Building the analyzer compiles (or loads) every model through the
# MIGraphX cache and runs the warmup inference
try:
	recognition.create_analyzer(config)
except FileNotFoundError:
	sys.exit(1)

pack = config.get("core", "model_pack", fallback="buffalo_s")
print(_("\nDone, compilation and warmup took %.1fs") % (time.time() - start, ))
print(_("Compiled models are stored in %s") % (paths_factory.compiled_models_dir_path(pack), ))
