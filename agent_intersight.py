# intersight_agent.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
# <<< NEW IMPORT FOR CORS
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import re # For checking key type

# Intersight SDK specific imports
import intersight # Main import
from intersight.api_client import ApiClient
from intersight.configuration import Configuration
from intersight.signing import HttpSigningConfiguration
from intersight.api.compute_api import ComputeApi
from intersight.api.cond_api import CondApi
from intersight.model.compute_physical_summary import ComputePhysicalSummary
from intersight.model.cond_alarm import CondAlarm
from intersight.exceptions import ApiException
from datetime import timedelta

# Load environment variables from .env file
load_dotenv()

# --- Intersight API Configuration ---
INTERSIGHT_API_KEY_ID = os.getenv("INTERSIGHT_API_KEY_ID")
INTERSIGHT_SECRET_KEY_FILE_PATH = os.getenv("INTERSIGHT_SECRET_KEY_FILE")
INTERSIGHT_API_BASE_PATH = os.getenv("INTERSIGHT_API_BASE_PATH", "https://intersight.com")

if not INTERSIGHT_API_KEY_ID or not INTERSIGHT_SECRET_KEY_FILE_PATH:
    print("Error: INTERSIGHT_API_KEY_ID and INTERSIGHT_SECRET_KEY_FILE must be set in .env file.")
    exit(1)

if not os.path.exists(INTERSIGHT_SECRET_KEY_FILE_PATH):
    print(f"Error: Secret key file not found at path: {INTERSIGHT_SECRET_KEY_FILE_PATH}")
    exit(1)

# Read the secret key from file
try:
    with open(INTERSIGHT_SECRET_KEY_FILE_PATH, 'r') as f:
        api_secret_key_string = f.read()
except Exception as e:
    print(f"Error reading secret key file: {e}")
    exit(1)

# Determine signing algorithm based on key type
signing_algorithm = None
if re.search('BEGIN RSA PRIVATE KEY', api_secret_key_string):
    signing_algorithm = intersight.signing.ALGORITHM_RSASSA_PKCS1v15
elif re.search('BEGIN EC PRIVATE KEY', api_secret_key_string):
    signing_algorithm = intersight.signing.ALGORITHM_ECDSA_MODE_DETERMINISTIC_RFC6979
else:
    print("Error: Could not determine key type (RSA or EC) from secret key file.")
    exit(1)

# Configure Intersight SDK using the new approach
try:
    signing_config = HttpSigningConfiguration(
        key_id=INTERSIGHT_API_KEY_ID,
        private_key_string=api_secret_key_string,
        signing_scheme=intersight.signing.SCHEME_HS2019,
        signing_algorithm=signing_algorithm,
        hash_algorithm=intersight.signing.HASH_SHA256,
        signed_headers=[
            intersight.signing.HEADER_REQUEST_TARGET,
            intersight.signing.HEADER_HOST,
            intersight.signing.HEADER_DATE,
            intersight.signing.HEADER_DIGEST,
        ],
        signature_max_validity=timedelta(seconds=300)
    )
except AttributeError as e:
    print(f"Error: AttributeError during HttpSigningConfiguration setup. Details: {e}")
    exit(1)


configuration = Configuration(
    host=INTERSIGHT_API_BASE_PATH,
    signing_info=signing_config
)

# Create API client
api_client = ApiClient(configuration)

# Initialize FastAPI app
app = FastAPI(
    title="Intersight AI Agent",
    description="An agent to interact with Cisco Intersight API, callable by a chatbot.",
    version="1.0.0"
)

# --- ADD CORS MIDDLEWARE ---
# This allows requests from any origin. For production, you might want to restrict origins.
# Example: origins = ["http://localhost:8000", "http://your-frontend-domain.com"]
origins = ["*"] # Allows all origins for now

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)
# --- END CORS MIDDLEWARE CONFIGURATION ---


# --- Pydantic Models for API Requests/Responses ---
class PowerStateRequest(BaseModel):
    power_state: str # "On" or "Off"

class ServerBasicInfo(BaseModel):
    moid: str
    name: str
    model: Optional[str] = None
    serial: Optional[str] = None
    management_ip: Optional[str] = None
    oper_power_state: Optional[str] = None
    admin_power_state: Optional[str] = None
    health: Optional[str] = None


class AlarmInfo(BaseModel):
    moid: str
    code: Optional[str] = None
    severity: Optional[str] = None
    description: Optional[str] = None
    affected_mo_display_name: Optional[str] = None
    last_transition_time: Optional[str] = None


# --- API Endpoints ---

@app.get("/agent/servers", response_model=List[ServerBasicInfo])
async def get_servers_list(
    name: Optional[str] = Query(None, description="Filter servers by name (contains, case-insensitive)"),
    model: Optional[str] = Query(None, description="Filter servers by model"),
    serial: Optional[str] = Query(None, description="Filter servers by serial number")
):
    try:
        compute_api_instance = ComputeApi(api_client)

        filter_parts = []
        if name:
            filter_parts.append(f"contains(Name,'{name}') or contains(Dn,'{name}')")
        if model:
            filter_parts.append(f"Model eq '{model}'")
        if serial:
            filter_parts.append(f"Serial eq '{serial}'")

        filter_str = " and ".join(filter_parts) if filter_parts else ""
        select_fields = "Moid,Name,Dn,Model,Serial,MgmtIpAddress,OperPowerState,AdminPowerState,Operability"

        if filter_str:
            api_response = compute_api_instance.get_compute_physical_summary_list(
                select=select_fields,
                filter=filter_str
            )
        else:
            api_response = compute_api_instance.get_compute_physical_summary_list(
                select=select_fields
            )

        servers = []
        if api_response.results:
            for server_summary in api_response.results:
                servers.append(ServerBasicInfo(
                    moid=server_summary.moid,
                    name=server_summary.name if hasattr(server_summary, 'name') and server_summary.name else server_summary.dn,
                    model=server_summary.model,
                    serial=server_summary.serial,
                    management_ip=server_summary.mgmt_ip_address,
                    oper_power_state=server_summary.oper_power_state,
                    admin_power_state=server_summary.admin_power_state,
                    health=server_summary.operability
                ))
        return servers
    except ApiException as e:
        print(f"Intersight API Exception when calling get_compute_physical_summary_list: {e}\n")
        error_detail = str(e.body)
        if e.body and isinstance(e.body, str):
            try:
                import json
                error_body_json = json.loads(e.body)
                if isinstance(error_body_json, dict) and "Reason" in error_body_json:
                    error_detail = error_body_json["Reason"]
                elif isinstance(error_body_json, list) and len(error_body_json) > 0 and isinstance(error_body_json[0], dict) and "Message" in error_body_json[0]:
                     error_detail = error_body_json[0]["Message"]
            except json.JSONDecodeError:
                pass
        raise HTTPException(status_code=e.status, detail=error_detail)
    except Exception as e:
        print(f"Unexpected error fetching servers: {e}\n")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.post("/agent/servers/{server_moid}/power-state", response_model=Dict[str, Any])
async def set_server_power_state(server_moid: str, request: PowerStateRequest):
    if request.power_state not in ["On", "Off"]:
        raise HTTPException(status_code=400, detail="Invalid power_state. Must be 'On' or 'Off'.")

    try:
        compute_api_instance = ComputeApi(api_client)
        server_power_config = ComputePhysicalSummary(admin_power_state=request.power_state)

        api_response = compute_api_instance.patch_compute_physical_summary(
            moid=server_moid,
            compute_physical_summary=server_power_config
        )
        return {
            "message": f"Successfully requested to set power state to '{request.power_state}' for server {server_moid}.",
            "moid": api_response.moid,
            "requested_admin_power_state": api_response.admin_power_state if hasattr(api_response, 'admin_power_state') else "N/A"
        }
    except ApiException as e:
        print(f"Intersight API Exception when calling patch_compute_physical_summary for power state: {e}\n")
        error_detail = str(e.body)
        if e.body and isinstance(e.body, str):
            try:
                import json
                error_body_json = json.loads(e.body)
                if isinstance(error_body_json, dict) and "Reason" in error_body_json:
                    error_detail = error_body_json["Reason"]
                elif isinstance(error_body_json, list) and len(error_body_json) > 0 and isinstance(error_body_json[0], dict) and "Message" in error_body_json[0]:
                     error_detail = error_body_json[0]["Message"]
            except json.JSONDecodeError:
                pass
        raise HTTPException(status_code=e.status, detail=error_detail)
    except Exception as e:
        print(f"Unexpected error setting server power state: {e}\n")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/agent/alarms", response_model=List[AlarmInfo])
async def get_alarms_list(
    severity: Optional[str] = Query(None, description="Filter alarms by severity (Critical, Warning, Info, Cleared)"),
    acknowledged: Optional[str] = Query(None, description="Filter by acknowledged state string ('true', 'false', 'acknowledged', 'unacknowledged')"),
    affected_mo_name: Optional[str] = Query(None, description="Filter alarms by affected object name (contains, case-insensitive)")
):
    try:
        cond_api_instance = CondApi(api_client)
        filter_parts = []
        if severity:
            filter_parts.append(f"Severity eq '{severity}'")

        if acknowledged is not None:
            ack_value = 'none'
            if acknowledged.lower() in ['true', 'acknowledged']:
                ack_value = 'true'
            elif acknowledged.lower() in ['false', 'unacknowledged']:
                ack_value = 'false'

            if ack_value != 'none':
                 filter_parts.append(f"Acknowledge eq {ack_value}")


        if affected_mo_name:
            filter_parts.append(f"contains(AffectedMoDisplayName,'{affected_mo_name}')")

        filter_str = " and ".join(filter_parts) if filter_parts else ""
        select_fields = "Moid,Code,Severity,Description,AffectedMoDisplayName,LastTransitionTime,Acknowledge"

        if filter_str:
            api_response = cond_api_instance.get_cond_alarm_list(
                select=select_fields,
                filter=filter_str,
                orderby="LastTransitionTime desc"
            )
        else:
            api_response = cond_api_instance.get_cond_alarm_list(
                select=select_fields,
                orderby="LastTransitionTime desc"
            )

        alarms = []
        if api_response.results:
            for alarm_data in api_response.results:
                alarms.append(AlarmInfo(
                    moid=alarm_data.moid,
                    code=alarm_data.code,
                    severity=alarm_data.severity,
                    description=alarm_data.description,
                    affected_mo_display_name=alarm_data.affected_mo_display_name,
                    last_transition_time=str(alarm_data.last_transition_time) if alarm_data.last_transition_time else None
                ))
        return alarms
    except ApiException as e:
        print(f"Intersight API Exception when calling get_cond_alarm_list: {e}\n")
        error_detail = str(e.body)
        if e.body and isinstance(e.body, str):
            try:
                import json
                error_body_json = json.loads(e.body)
                if isinstance(error_body_json, dict) and "Reason" in error_body_json:
                    error_detail = error_body_json["Reason"]
                elif isinstance(error_body_json, list) and len(error_body_json) > 0 and isinstance(error_body_json[0], dict) and "Message" in error_body_json[0]:
                     error_detail = error_body_json[0]["Message"]
            except json.JSONDecodeError:
                pass
        raise HTTPException(status_code=e.status, detail=error_detail)
    except Exception as e:
        print(f"Unexpected error fetching alarms: {e}\n")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    print("Starting Intersight Agent on http://localhost:8002")
    uvicorn.run(app, host="0.0.0.0", port=8002)
    