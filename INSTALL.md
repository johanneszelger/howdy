# Installing Howdy (ONNX edition) from source

Complete procedure for a fresh machine, as used on Ubuntu 26.04 with an AMD
Ryzen AI 9 HX 370 (Radeon 890M iGPU, gfx1150). CPU-only machines can skip the
ROCm parts (marked **GPU only**).

## 1. Build dependencies

```sh
sudo apt update && sudo apt install -y \
    python3 meson ninja-build build-essential cmake \
    libpam0g-dev libinih-dev libevdev-dev pkg-config \
    curl unzip
```

## 2. Python virtualenv at /opt/howdy/.venv

The venv path gets baked into the PAM module at build time — keep it stable.
The AMD GPU onnxruntime wheel only exists for specific Python versions
(3.10/3.12), so pin 3.12 via [uv](https://docs.astral.sh/uv/):

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
sudo mkdir -p /opt/howdy && sudo chown "$USER" /opt/howdy
~/.local/bin/uv venv --seed --python 3.12 /opt/howdy/.venv
/opt/howdy/.venv/bin/pip install numpy opencv-python insightface
```

Consider `sudo chown -R root:root /opt/howdy/.venv` at the end of the install:
root executes this interpreter during authentication, so it should not stay
user-writable.

### onnxruntime

**CPU only:**

```sh
/opt/howdy/.venv/bin/pip install onnxruntime
```

**GPU only (AMD):** install the MIGraphX build instead — it must match the
ROCm version installed in step 3:

```sh
/opt/howdy/.venv/bin/pip install \
    https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/onnxruntime_migraphx-1.23.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```

## 3. ROCm runtime (GPU only)

AMD's apt repo (the `noble` line works on newer Ubuntu; packages are
self-contained under /opt/rocm-<version>):

```sh
sudo mkdir -p /etc/apt/keyrings
wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.2.4 noble main" | sudo tee /etc/apt/sources.list.d/rocm.list
printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' | sudo tee /etc/apt/preferences.d/rocm-pin-600
sudo apt update
sudo apt install migraphx rocminfo rocm-hip-runtime
```

Do NOT install `amdgpu-dkms` — the mainline kernel driver is newer.

Make the libraries visible to PAM-spawned processes (no LD_LIBRARY_PATH at
login time):

```sh
echo /opt/rocm/lib | sudo tee /etc/ld.so.conf.d/rocm.conf && sudo ldconfig
```

Verify the GPU is visible: `/opt/rocm/bin/rocminfo | grep gfx` should list
your architecture (e.g. `gfx1150`).

## 4. Build and install Howdy

```sh
git clone <your-fork> howdy && cd howdy && git checkout onnx-migration
meson setup build -Dpython_path=/opt/howdy/.venv/bin/python3
meson compile -C build
sudo meson install -C build
```

Installs to /usr/local: python sources in
`/usr/local/lib/<arch>/howdy/`, config in `/usr/local/etc/howdy/config.ini`
(only written if missing), `pam_howdy.so` in
`/usr/local/lib/<arch>/security/`, data dir `/usr/local/share/dlib-data/`.

## 5. Face model pack

```sh
cd /usr/local/share/dlib-data && sudo ./install.sh          # buffalo_s (default)
```

## 6. Configuration

`sudo howdy config` (or edit `/usr/local/etc/howdy/config.ini`). The
important values:

```ini
[core]
model_pack = buffalo_s
execution_provider = auto   # auto picks the GPU when available, else CPU
det_size = 640              # >= frame width processes native camera resolution
det_thresh = 0.5

[video]
similarity_threshold = 0.6  # cosine similarity, higher = stricter (see below)
device_path = /dev/v4l/by-path/<your-IR-camera>   # find via: ls /dev/v4l/by-path/
max_height = 400            # camera native height (check: howdy test)
dark_threshold = 80
timeout = 2
```

## 7. Compile the models (GPU only) and enroll

```sh
sudo howdy compile     # one-time MIGraphX compile, a few minutes; cached afterwards
sudo howdy add         # enroll your face
sudo howdy test        # live window: green circle + similarity score, provider name
```

Threshold tuning: `howdy test` prints per-second console lines and
MATCH/NO MATCH transitions with the live cosine score. Same-person scores are
typically 0.7+, different-person ~0.0-0.2, so 0.6 is strict yet comfortable.
Enroll extra models (`howdy add`, label them) for glasses/no-glasses etc.

## 8. PAM wiring

Add howdy to the PAM services you want (e.g. `/etc/pam.d/sudo`,
`/etc/pam.d/sddm`, KDE: `/etc/pam.d/kde`), as the FIRST auth line:

```
auth    sufficient    /usr/local/lib/x86_64-linux-gnu/security/pam_howdy.so
```

`sufficient` keeps password fallback working. Keep a root shell open in a
second terminal while testing:

```sh
sudo /opt/howdy/.venv/bin/python3 /usr/local/lib/x86_64-linux-gnu/howdy/compare.py "$USER"; echo "exit: $?"   # 0 = success
sudo -k && sudo echo it-works                                                  # live test
```

## Known quirk (AMD APUs)

The MIGraphX execution provider can return stale output buffers on APUs
(missing stream synchronization; seen on gfx1150 with onnxruntime 1.23.2).
Howdy sets `HIP_LAUNCH_BLOCKING=1` automatically as a workaround — nothing to
do, just don't remove that code from `recognition.py`.

## Uninstall / rollback

```sh
sudo ninja -C build uninstall     # or remove the installed files listed above
sudo rm /etc/pam.d/...            # remove the pam_howdy.so lines you added
```
