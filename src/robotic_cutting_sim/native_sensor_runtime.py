"""Opt-in Isaac Sim 6.2 / Kit 110.4 offline contact-extension activation.

The installed official registry caches contain the metadata the dependency
solver needs, but Kit 110.4 checks HTTP reachability before using those caches.
This profile presents identical metadata through episode-owned file registries.
It changes neither extension manifests nor installed extension payloads.
"""
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath

from .recording import write_json


PROFILE = "isaac62_physx_offline_v1"
REGISTRIES_KEY = "/exts/omni.kit.registry.nucleus/registries"
# Official cached indices from the installed Kit 110.4.0+187cf21b distribution.
# Pin the registry maps; each map in turn verifies all its JSON records by SHA1.
INDICES = (
    ("3acb1cf9", "kit/prod/default", "kit-default",
     "6f9a593fb38fe977eb7b914785778aaa2a13d1144e539254fa9c135ee96ae759"),
    ("c403fc51", "kit/prod/sdk", "kit-sdk",
     "0f70878f96f8a68b599a27309ffbb4c63b9ada19cd71b7e059df1356039893ec"),
)
SENSOR_EXTENSION = "isaacsim.sensors.experimental.physics"


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _store_identical(path, data):
    """Never replace a different cache file, including after a partial run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise ValueError(f"Offline registry file already differs: {path}")


def prepare_offline_registry(output, *, cache_root=None):
    """Copy verified generic metadata only; safe before importing any Kit module."""
    output = Path(output).resolve()
    cache_root = (Path.home()/".local/share/ov/data/exts/v2/index"
                  if cache_root is None else Path(cache_root))
    mirror = output/"registry"
    rows, registries = [], []
    for cache_id, name, label, registry_sha256 in INDICES:
        source = cache_root/cache_id
        raw_registry = (source/"registry.json").read_bytes()
        if _sha256(raw_registry) != registry_sha256:
            raise ValueError(f"Unexpected official Kit 110.4 registry map: {source}")
        entries = {"registry.json": None, **json.loads(raw_registry)["files"]}
        for relative, sha1_prefix in sorted(entries.items()):
            relative_path = PurePosixPath(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts or relative_path.suffix != ".json":
                raise ValueError(f"Invalid registry metadata path: {relative!r}")
            raw = raw_registry if relative == "registry.json" else (source/relative).read_bytes()
            if sha1_prefix is not None and not hashlib.sha1(raw).hexdigest().startswith(sha1_prefix):
                raise ValueError(f"Official registry metadata hash mismatch: {source/relative}")
            compressed = gzip.compress(raw, mtime=0)
            target = mirror/label/"v2"/relative_path.with_suffix(".gz")
            _store_identical(target, compressed)
            rows.append(dict(registry=name, source=str(source/relative),
                relative_path=relative, bytes=len(raw), sha256=_sha256(raw),
                mirror_path=str(target.relative_to(output)), gzip_sha256=_sha256(compressed)))
        registries.append(dict(name=name, url=(mirror/label).as_uri(), trusted=True))
    receipt = dict(profile=PROFILE, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        status="PASS_METADATA_INTEGRITY", files=rows, registries=registries,
        installed_extension_payloads="UNCHANGED", native_sensor_updates="NOT_RUN")
    write_json(output/"metadata.json", receipt)
    return registries, receipt


def activate_offline_contact_extension(app, output):
    """Apply real process settings, refresh their providers, and enable the sensor.

    Activation is separate from valid post-physics sensor updates. The latter
    remain measured by NativeToolContacts and its original physical trace.
    """
    import carb.eventdispatcher
    import carb.settings
    import omni.kit.registry.nucleus as registry
    from .runtime_memory import record_memory

    output = Path(output)/"native_sensor_runtime_v1"
    output.mkdir(parents=True, exist_ok=True)
    receipt = dict(profile=PROFILE, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                   status="ACTIVATING", native_sensor_updates="NOT_RUN")
    try:
        receipt["isaacsim_version"] = importlib.metadata.version("isaacsim")
        receipt["kit_version"] = app.get_kit_version()
        kit_version = receipt["kit_version"]
        if (receipt["isaacsim_version"] != "6.2.0.0" or
                kit_version.split("+", 1)[0] != "110.4.0" or "187cf21b" not in kit_version):
            raise RuntimeError(f"{PROFILE} requires Isaac Sim 6.2.0.0 / Kit 110.4.0 build 187cf21b")
        registries, metadata = prepare_offline_registry(output)
        receipt["metadata_file_count"] = len(metadata["files"])
        settings = carb.settings.get_settings()
        # user.config.json loads /persistent only. Set the real nonpersistent
        # registry array and notify the installed provider, not the user file.
        receipt["previous_registries"] = settings.get_settings_dictionary("").get(REGISTRIES_KEY[1:], [])
        settings.set(REGISTRIES_KEY, registries)
        readback = settings.get_settings_dictionary("").get(REGISTRIES_KEY[1:], [])
        if readback != registries:
            raise RuntimeError("Kit did not accept the offline registry settings")
        receipt["configured_registries"] = readback
        carb.eventdispatcher.get_eventdispatcher().dispatch_event(
            event_name="omni.kit.registry.nucleus.REGISTRIES_CHANGED_EVENT")
        # Read through the real provider, not just the settings store. A fresh
        # provider selects its v2 implementation only when sync_index() reads
        # the index. This is local metadata I/O, not a physics/app update.
        provider_counts = {}
        registry_manager = registry.get_registry_manager_instance()
        for row in registries:
            provider = registry_manager.get_provider(row["name"])
            index = provider.sync_index() if provider is not None else None
            if index is None or not len(index):
                raise RuntimeError(f"Kit did not read the local registry index: {row['name']}")
            provider_counts[row["name"]] = len(index)
        receipt["provider_index_record_counts"] = provider_counts
        urls = {row["name"]: registry.get_registry_url_by_name(row["name"]) for row in registries}
        receipt["provider_urls"] = urls
        if any(urls[row["name"]] != row["url"]+"/v2" for row in registries):
            raise RuntimeError("Kit registry provider did not consume the offline URLs")
        manager = app.get_extension_manager()
        enabled = manager.set_extension_enabled_immediate(SENSOR_EXTENSION, True)
        receipt["activation_returned"] = bool(enabled)
        if not enabled:
            raise RuntimeError(f"Kit could not activate {SENSOR_EXTENSION}; see kit.log")
        closure = {}
        for name in (SENSOR_EXTENSION, "isaacsim.robot.schema", "omni.physics", "omni.physics.physx",
                     "omni.physx.tensors", "isaacsim.physics.newton.tensors", "isaacsim.test.utils"):
            extension_id = manager.get_enabled_extension_id(name)
            if not extension_id:
                closure[name] = dict(enabled=False)
                continue
            manifest = Path(manager.get_extension_path(extension_id))/"config/extension.toml"
            closure[name] = dict(enabled=True, extension_id=extension_id,
                version=manager.get_extension_dict(extension_id)["package"]["version"],
                manifest=str(manifest), manifest_sha256=_sha256(manifest.read_bytes()))
        receipt["enabled_extensions"] = closure
        if closure[SENSOR_EXTENSION]["version"] != "4.0.0":
            raise RuntimeError("The offline profile requires the installed contact sensor extension 4.0.0")
        # Import the actual enabled module; constructing/stepping its native
        # sensor remains mandatory in reset() and is never replaced by this.
        from isaacsim.sensors.experimental.physics import Contact, ContactSensor
        receipt["module_classes"] = [Contact.__module__, ContactSensor.__module__]
        receipt["status"] = "PASS_EXTENSION_ENABLED"
    except BaseException as error:
        receipt.update(status="FAIL_EXTENSION_ACTIVATION", error=f"{type(error).__name__}: {error}")
        write_json(output/"activation.json", receipt)
        raise
    write_json(output/"activation.json", receipt)
    record_memory(os.environ.get("RCS_RUNTIME_MEMORY_LOG"), "after_contact_extension_activation",
                  sensor_extension=SENSOR_EXTENSION, runtime_profile=PROFILE)
    return receipt
