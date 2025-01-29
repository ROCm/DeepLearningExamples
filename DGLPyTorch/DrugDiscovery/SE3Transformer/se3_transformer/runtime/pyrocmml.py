import amdsmi
import subprocess

cpu_node_dict = {}

def rocmmlInit():
    amdsmi.amdsmi_init()

def rocmDeviceGetHandleByIndex(device_idx):
    try:
        devices = amdsmi.amdsmi_get_processor_handles()
        if len(devices) == 0:
            print("No GPUs on machine")

        return devices[device_idx]                       
    except amdsmi.AmdSmiException as e:
        print(e)

def rocmDeviceGetName(handle):
    try: 
        device_asic_info = amdsmi.amdsmi_get_gpu_asic_info(handle)
        device_id = device_asic_info["device_id"]
        device_vendor_name = device_asic_info["vendor_name"]
        return device_vendor_name + " " + str(device_id)
    except amdsmi.AmdSmiException as e:
        print(e)

def rocmDeviceGetUUID(handle):
    try:
        uuid = amdsmi.amdsmi_get_gpu_device_uuid(handle)
        return uuid
    except amdsmi.AmdSmiException as e:
        print(e)

def get_cpu_node_dict():
    try: 
        result = subprocess.run(["lscpu", "-e=CPU,NODE"], capture_output=True, text=True)
        result.check_returncode()  # Raise an exception for non-zero exit codes

        for line in result.stdout.splitlines()[1:]:
            parts = line.strip().split()
            if len(parts) == 2:  # Ensure we have CPU and NODE
                cpu_id, node_id = parts
                if int(node_id) not in cpu_node_dict:
                    cpu_node_dict[int(node_id)] = []
                cpu_node_dict[int(node_id)].append(int(cpu_id))

    except subprocess.CalledProcessError as e:
        print("Error executing 'lscpu': {e}")
        return {}

def rocmDeviceGetCpuAffinityWithinScope(handle):
    if len(cpu_node_dict)==0:
        get_cpu_node_dict()
    try: 
        numa_node = amdsmi.amdsmi_get_gpu_topo_numa_affinity(handle)
        numa_list = cpu_node_dict[numa_node]
        return numa_list
    except amdsmi.AmdSmiException as e:
        print(e)

