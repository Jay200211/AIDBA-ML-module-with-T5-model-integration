"""Test the approval workflow state machine."""
import sys
import time
import requests

BASE_URL = "http://localhost:8000"

def wait_for_server():
    """Wait until AIDBA is ready."""
    print("⏳ Waiting for AIDBA to be ready...")
    for i in range(30):
        try:
            r = requests.get(f"{BASE_URL}/api/health", timeout=2)
            if r.status_code == 200:
                print("✅ AIDBA is ready!")
                return True
        except:
            pass
        time.sleep(1)
    print("❌ AIDBA did not start in 30 seconds")
    return False

def test_state_machine():
    if not wait_for_server():
        return

    print("\n=== 1. Creating Test Proposal ===")
    r = requests.post(f"{BASE_URL}/api/proposals/create_test")
    print(f"Response: {r.status_code} - {r.text}")
    if r.status_code != 200:
        print("❌ Failed to create proposal")
        return
    data = r.json()
    if not data.get("ok"):
        print(f"❌ Error: {data.get('error', 'Unknown')}")
        return
    proposal_id = data["id"]
    print(f"✅ Created proposal: {proposal_id}")

    print("\n=== 2. Checking Allowed Transitions ===")
    r = requests.get(f"{BASE_URL}/api/proposals/{proposal_id}/allowed")
    print(f"Response: {r.status_code}")
    print(f"Current state: {r.json().get('current_state')}")
    print(f"Allowed next: {r.json().get('allowed_transitions')}")

    print("\n=== 3. Running Through State Machine ===")
    transitions = ['Reviewed', 'Approved', 'Testing', 'Deploying', 'Monitoring', 'Completed']
    for state in transitions:
        r = requests.post(
            f"{BASE_URL}/api/proposals/{proposal_id}/transition",
            json={"state": state, "approver": "dba_user", "comment": f"Moving to {state}"}
        )
        if r.status_code == 200:
            data = r.json()
            print(f"✅ {data['from']} → {data['new_state']}")
        else:
            print(f"❌ Error at {state}: {r.text}")
            break

    print("\n=== 4. Final State ===")
    r = requests.get(f"{BASE_URL}/api/proposals")
    proposals = r.json().get("proposals", [])
    final = next((p for p in proposals if p["id"] == proposal_id), None)
    if final:
        print(f"Final state: {final['state']}")
        print(f"Approver: {final['approver']}")
        print(f"Comment: {final.get('comment', 'None')}")
    else:
        print("❌ Proposal not found")

    print("\n=== 5. Transition History ===")
    r = requests.get(f"{BASE_URL}/api/proposals/{proposal_id}/history")
    history = r.json().get("history", [])
    print(f"Total history events: {len(history)}")
    for event in history:
        print(f"  - {event['ts']}: {event['event_type']}")

    print("\n✅ Test complete!")

if __name__ == "__main__":
    try:
        test_state_machine()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
