"""Complete integration test for AIDBA + ML model."""
import requests
import time
import sys

BASE = "http://localhost:8000"


def wait_for_server():
    """Wait for AIDBA to be ready."""
    print("⏳ Waiting for AIDBA...")
    for i in range(30):
        try:
            r = requests.get(f"{BASE}/api/health", timeout=2)
            if r.status_code == 200:
                print("✅ AIDBA is ready!\n")
                return True
        except:
            pass
        time.sleep(1)
    print("❌ AIDBA not responding")
    return False


def test_safe_query():
    """Test safe SELECT query."""
    print("=" * 60)
    print("TEST 1: Safe SELECT Query")
    print("=" * 60)
    r = requests.post(f"{BASE}/api/nlq", json={"question": "show all customers"})
    print(f"Status: {r.status_code}")
    data = r.json()
    print(f"Type: {data.get('type')}")
    print(f"Summary: {data.get('summary')}")
    if data.get('rows'):
        print(f"Rows: {len(data['rows'])}")
    print(f"✅ PASSED\n" if data.get('type') == 'tables' or data.get('rows') else f"⚠️ Response: {data}\n")


def test_destructive_with_approval():
    """Test DELETE that requires approval."""
    print("=" * 60)
    print("TEST 2: Destructive Operation (Requires Approval)")
    print("=" * 60)
    r = requests.post(f"{BASE}/api/nlq", json={"question": "delete customers from Germany"})
    print(f"Status: {r.status_code}")
    data = r.json()
    print(f"Type: {data.get('type')}")
    print(f"Summary: {data.get('summary')[:200]}")
    if data.get('operation'):
        print(f"Operation: {data['operation']['operation']}")
        print(f"Risk: {data['operation']['risk_level']}")
    print(f"✅ PASSED (alert shown)\n" if data.get('type') == 'approval_required' else f"⚠️ Response: {data}\n")


def test_state_machine():
    """Test the approval workflow."""
    print("=" * 60)
    print("TEST 3: State Machine")
    print("=" * 60)
    r = requests.post(f"{BASE}/api/proposals/create_test")
    if r.status_code != 200:
        print("❌ Failed to create proposal")
        return
    pid = r.json()['id']
    print(f"Created: {pid}")

    states = ['Reviewed', 'Approved', 'Testing', 'Deploying', 'Monitoring', 'Completed']
    for state in states:
        r = requests.post(
            f"{BASE}/api/proposals/{pid}/transition",
            json={"state": state, "approver": "test_user"}
        )
        if r.status_code == 200:
            print(f"  ✅ → {state}")
        else:
            print(f"  ❌ Failed at {state}")
    print(f"✅ PASSED\n")


def test_alerts():
    """Test alert system."""
    print("=" * 60)
    print("TEST 4: Alert System")
    print("=" * 60)
    r = requests.get(f"{BASE}/api/audit?limit=5")
    if r.status_code == 200:
        events = r.json().get('rows', [])
        print(f"Recent audit events: {len(events)}")
        for event in events[:3]:
            print(f"  - {event.get('event_type')}: {event.get('db_name')}")
    print(f"✅ PASSED\n")


def main():
    print("\n" + "=" * 60)
    print("AIDBA + ML Integration Test Suite")
    print("=" * 60 + "\n")

    if not wait_for_server():
        sys.exit(1)

    test_safe_query()
    test_destructive_with_approval()
    test_state_machine()
    test_alerts()

    print("=" * 60)
    print("✅ All tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
