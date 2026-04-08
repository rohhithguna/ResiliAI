#!/usr/bin/env python3
"""
Phase 2 LiteLLM Proxy Validator Simulation
Simulates the strict Phase 2 validation protocol
"""

import subprocess
import time
import json
import sys
import os
from pathlib import Path

# Test configuration
API_BASE_URL = "https://proxy-validator.test/v1"
API_KEY = "validator-test-key"
PORT = 9999
HOST = "127.0.0.1"

def log(msg, level="INFO"):
    print(f"[{level}] {msg}")

def run_inference():
    """Simulate running inference.py directly (SUBMISSION PATH)"""
    log("TESTING: python3 inference.py (SUBMISSION PATH)")
    env = os.environ.copy()
    env["API_BASE_URL"] = API_BASE_URL
    env["API_KEY"] = API_KEY
    env["MODEL_NAME"] = "gpt-3.5-turbo"
    
    result = subprocess.run(
        ["python3", "inference.py"],
        cwd="/Users/rohhithg/Desktop/meta_project",
        capture_output=True,
        text=True,
        timeout=30,
        env=env
    )
    
    log(f"Return code: {result.returncode}")
    if "[START]" in result.stdout and "[END]" in result.stdout:
        log("✅ Execution markers found: [START]...[END]")
        return True
    else:
        log(f"❌ Missing execution markers. Output:\n{result.stdout[:200]}\n{result.stderr[:200]}")
        return False

def test_api_endpoint():
    """Simulate API /run endpoint (HF DEPLOYMENT PATH)"""
    log("TESTING: POST /run endpoint (HF DEPLOYMENT PATH)")
    
    # Start server
    import subprocess
    env = os.environ.copy()
    env["API_BASE_URL"] = API_BASE_URL
    env["API_KEY"] = API_KEY
    env["MODEL_NAME"] = "gpt-3.5-turbo"
    
    try:
        # Start uvicorn in background
        server = subprocess.Popen(
            ["python3", "-m", "uvicorn", "api:app", f"--host={HOST}", f"--port={PORT}"],
            cwd="/Users/rohhithg/Desktop/meta_project",
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        time.sleep(4)
        
        # Test endpoints
        endpoints = [
            ("GET", f"http://{HOST}:{PORT}/", "root"),
            ("GET", f"http://{HOST}:{PORT}/health", "health"),
            ("POST", f"http://{HOST}:{PORT}/run", "run"),
        ]
        
        all_ok = True
        for method, url, name in endpoints:
            cmd = ["curl", "-s", "-X", method, url, "-H", "Content-Type: application/json"]
            if method == "POST":
                cmd.extend(["-d", '{"task": "easy"}'])
            cmd.extend(["-w", "\n%{http_code}"])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')
            http_code = lines[-1] if lines else "ERROR"
            
            if http_code == "200":
                log(f"✅ {name} ({method}): {http_code}")
            else:
                log(f"❌ {name} ({method}): {http_code}", "ERROR")
                all_ok = False
        
        return all_ok
    finally:
        server.terminate()
        server.wait(timeout=5)

def verify_env_access():
    """Verify that code uses strict os.environ[] access"""
    log("TESTING: Strict environment variable access (no fallbacks)")
    
    inference_file = Path("/Users/rohhithg/Desktop/meta_project/src/inference/inference.py").read_text()
    root_inference = Path("/Users/rohhithg/Desktop/meta_project/inference.py").read_text()
    
    checks = [
        ('os.environ["API_BASE_URL"]', "API_BASE_URL strict access"),
        ('os.environ["API_KEY"]', "API_KEY strict access"),
        ("call_llm", "call_llm function exists"),
    ]
    
    all_ok = True
    for pattern, desc in checks:
        if pattern in inference_file and pattern in root_inference:
            log(f"✅ {desc}: Found in both inference paths")
        else:
            log(f"❌ {desc}: Missing", "ERROR")
            all_ok = False
    
    return all_ok

def main():
    log("=" * 60)
    log("PHASE 2 LITEMLLM PROXY VALIDATOR SIMULATION")
    log("=" * 60)
    
    results = {}
    
    # Test 1: Environment access verification
    log("\n[TEST 1/4] Environment variable access pattern")
    results["env_access"] = verify_env_access()
    
    # Test 2: Inference.py execution (submission path)
    log("\n[TEST 2/4] Submission path: python3 inference.py")
    try:
        results["inference_path"] = run_inference()
    except Exception as e:
        log(f"❌ Exception: {e}", "ERROR")
        results["inference_path"] = False
    
    # Test 3: API /run endpoint (HF deployment path)
    log("\n[TEST 3/4] HF deployment path: POST /run")
    try:
        results["api_path"] = test_api_endpoint()
    except Exception as e:
        log(f"❌ Exception: {e}", "ERROR")
        results["api_path"] = False
    
    # Final result
    log("\n" + "=" * 60)
    log("VALIDATION RESULTS")
    log("=" * 60)
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        log(f"{test}: {status}")
    
    all_passed = all(results.values())
    log("\n" + "=" * 60)
    if all_passed:
        log("FINAL RESULT: PASS ✅", "SUCCESS")
        log("All Phase 2 strict validation checks passed")
    else:
        log("FINAL RESULT: FAIL ❌", "ERROR")
    log("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
