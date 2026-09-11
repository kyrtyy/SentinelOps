"""
lambda_function.py

AWS Lambda function triggered by EventBridge on CloudWatch Alarms or SRE incident events.
Extracts incident details and dispatches a triage request to the SentinelOps model server.
"""
import json
import os
import urllib.request

MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://10.0.10.100:8000/v1/chat/completions")


def lambda_handler(event, context):
    print("Received incident event:", json.dumps(event))

    # Extract alarm details from CloudWatch alarm or custom event payload
    detail = event.get("detail", {})
    alarm_name = detail.get("alarmName", event.get("alarm_name", "ProductionIncidentAlarm"))
    state = detail.get("state", {}).get("value", event.get("state", "ALARM"))
    reason = detail.get("state", {}).get("reason", event.get("reason", "Metric threshold breached"))

    prompt = (
        f"Incident Alarm Triggered: `{alarm_name}` in state `{state}`.\n\n"
        f"Reason / Alarm Description:\n```\n{reason}\n```\n\n"
        "Diagnose the root cause and take the appropriate remediation action."
    )

    payload = {
        "model": "sentinelops",
        "messages": [
            {
                "role": "system",
                "content": "You are SentinelOps, an autonomous SRE incident-response agent. Investigate using tools and remediate."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1
    }

    try:
        req = urllib.request.Request(
            MODEL_SERVER_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            print("SentinelOps Triage Response:", json.dumps(result))
            return {
                "statusCode": 200,
                "body": json.dumps({"status": "triage_initiated", "response": result})
            }
    except Exception as e:
        print(f"Error calling SentinelOps model server: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
