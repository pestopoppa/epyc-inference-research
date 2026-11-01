import ctypes as c
import json

lib = c.CDLL("/opt/rocm-6.2.0/lib/librocm_smi64.so.7")
class Counter(c.Structure):
    _fields_ = [("type", c.c_int), ("value", c.c_uint64)]
lib.rsmi_init.argtypes = [c.c_uint64]
lib.rsmi_num_monitor_devices.argtypes = [c.POINTER(c.c_uint32)]
lib.rsmi_dev_pci_id_get.argtypes = [c.c_uint32, c.POINTER(c.c_uint64)]
lib.rsmi_utilization_count_get.argtypes = [c.c_uint32, c.POINTER(Counter), c.c_uint32, c.POINTER(c.c_uint64)]
lib.rsmi_shut_down.argtypes = []
for name in ("rsmi_init", "rsmi_num_monitor_devices", "rsmi_dev_pci_id_get", "rsmi_utilization_count_get", "rsmi_shut_down"):
    getattr(lib, name).restype = c.c_int
report = {"library": "/opt/rocm-6.2.0/lib/librocm_smi64.so.7", "devices": []}
report["init_rc"] = lib.rsmi_init(0)
if report["init_rc"] == 0:
    try:
        count = c.c_uint32()
        report["enumerate_rc"] = lib.rsmi_num_monitor_devices(c.byref(count))
        if report["enumerate_rc"] == 0:
            for index in range(count.value):
                bdf = c.c_uint64()
                rc = lib.rsmi_dev_pci_id_get(index, c.byref(bdf))
                report["devices"].append({"index": index, "pci_rc": rc, "bdfid": hex(bdf.value) if rc == 0 else None})
            matching = [row for row in report["devices"] if row["bdfid"] == "0x4300"]
            if len(matching) == 1:
                counter = Counter(0, 0)
                timestamp = c.c_uint64()
                rc = lib.rsmi_utilization_count_get(matching[0]["index"], c.byref(counter), 1, c.byref(timestamp))
                report["counter"] = {"rc": rc, "type": counter.type,
                    "value": counter.value if rc == 0 else None,
                    "timestamp_ns": timestamp.value if rc == 0 else None}
    finally:
        report["shutdown_rc"] = lib.rsmi_shut_down()
print(json.dumps(report, sort_keys=True))
