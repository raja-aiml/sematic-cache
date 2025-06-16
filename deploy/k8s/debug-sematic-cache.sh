#!/bin/bash
set -euo pipefail

NAMESPACE="app"
APP_LABEL="app=sematic-cache"

echo "🔍 Searching for pods with label '$APP_LABEL' in namespace '$NAMESPACE'..."

# Check if namespace exists
if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
  echo "❌ Namespace '$NAMESPACE' does not exist"
  exit 1
fi

# Get pod information
PODS=$(kubectl get pods -n "$NAMESPACE" -l "$APP_LABEL" -o json 2>/dev/null || echo '{"items":[]}')

if ! echo "$PODS" | jq -e '.items | length > 0' >/dev/null 2>&1; then
  echo "❌ No pods found with label '$APP_LABEL' in namespace '$NAMESPACE'"
  echo "📦 All pods in '$NAMESPACE' namespace:"
  kubectl get pods -n "$NAMESPACE" 2>/dev/null || echo "No pods found"
  exit 1
fi

# Get the most recent pod (by creation time)
POD=$(echo "$PODS" | jq -r '.items | sort_by(.metadata.creationTimestamp) | reverse | .[0].metadata.name')

if [[ -z "${POD:-}" || "$POD" == "null" ]]; then
  echo "❌ Failed to get pod name"
  exit 1
fi

echo "🎯 Selected pod: $POD"

# Display pods overview
echo -e "\n📦 Pods in '$NAMESPACE' namespace:"
kubectl get pods -n "$NAMESPACE"

# Get detailed pod information
echo -e "\n📊 Pod Status Summary:"
POD_INFO=$(kubectl get pod -n "$NAMESPACE" "$POD" -o json)

# Extract key information safely
PHASE=$(echo "$POD_INFO" | jq -r '.status.phase // "Unknown"')
READY=$(echo "$POD_INFO" | jq -r '.status.conditions[] | select(.type=="Ready") | .status // "Unknown"')
RESTART_COUNT=$(echo "$POD_INFO" | jq -r '.status.containerStatuses[0].restartCount // 0')
IMAGE=$(echo "$POD_INFO" | jq -r '.spec.containers[0].image // "Unknown"')

echo "🆔 Pod Name:       $POD"
echo "📊 Phase:          $PHASE"
echo "✅ Ready:          $READY"
echo "🔁 Restart Count:  $RESTART_COUNT"
echo "🐳 Image:          $IMAGE"

# Container status details
if echo "$POD_INFO" | jq -e '.status.containerStatuses[0]' >/dev/null 2>&1; then
  CONTAINER_STATUS=$(echo "$POD_INFO" | jq -r '.status.containerStatuses[0]')
  CURRENT_STATE=$(echo "$CONTAINER_STATUS" | jq -r '.state | keys[0]')
  
  echo "📌 Current State:  $CURRENT_STATE"
  
  # Show current state details
  case "$CURRENT_STATE" in
    "waiting")
      REASON=$(echo "$CONTAINER_STATUS" | jq -r '.state.waiting.reason // "Unknown"')
      MESSAGE=$(echo "$CONTAINER_STATUS" | jq -r '.state.waiting.message // ""')
      echo "⏳ Waiting Reason: $REASON"
      [[ -n "$MESSAGE" && "$MESSAGE" != "null" ]] && echo "💬 Message:        $MESSAGE"
      ;;
    "running")
      STARTED=$(echo "$CONTAINER_STATUS" | jq -r '.state.running.startedAt // "Unknown"')
      echo "🏃 Started At:     $STARTED"
      ;;
    "terminated")
      EXIT_CODE=$(echo "$CONTAINER_STATUS" | jq -r '.state.terminated.exitCode // "Unknown"')
      REASON=$(echo "$CONTAINER_STATUS" | jq -r '.state.terminated.reason // "Unknown"')
      MESSAGE=$(echo "$CONTAINER_STATUS" | jq -r '.state.terminated.message // ""')
      echo "💥 Exit Code:      $EXIT_CODE"
      echo "📖 Reason:         $REASON"
      [[ -n "$MESSAGE" && "$MESSAGE" != "null" ]] && echo "💬 Message:        $MESSAGE"
      ;;
  esac
  
  # Show last state if container has been restarted
  if echo "$CONTAINER_STATUS" | jq -e '.lastState.terminated' >/dev/null 2>&1; then
    echo -e "\n🔄 Last Termination:"
    LAST_EXIT_CODE=$(echo "$CONTAINER_STATUS" | jq -r '.lastState.terminated.exitCode // "Unknown"')
    LAST_REASON=$(echo "$CONTAINER_STATUS" | jq -r '.lastState.terminated.reason // "Unknown"')
    LAST_MESSAGE=$(echo "$CONTAINER_STATUS" | jq -r '.lastState.terminated.message // ""')
    echo "💥 Last Exit Code: $LAST_EXIT_CODE"
    echo "📖 Last Reason:    $LAST_REASON"
    [[ -n "$LAST_MESSAGE" && "$LAST_MESSAGE" != "null" ]] && echo "💬 Last Message:   $LAST_MESSAGE"
  fi
fi

# Show recent events
echo -e "\n📅 Recent Events:"
kubectl get events -n "$NAMESPACE" --field-selector involvedObject.name="$POD" --sort-by='.lastTimestamp' | tail -10

# Try to get logs
echo -e "\n📜 Container Logs:"
if kubectl logs -n "$NAMESPACE" "$POD" --tail=50 2>/dev/null; then
  echo "✅ Logs retrieved successfully"
else
  echo "⚠️ Could not fetch current logs, trying previous container..."
  if kubectl logs -n "$NAMESPACE" "$POD" --previous --tail=50 2>/dev/null; then
    echo "✅ Previous container logs retrieved"
  else
    echo "❌ No logs available"
  fi
fi

# Advanced debugging for binary issues
echo -e "\n🔍 DEEP DIVE ANALYSIS:"

# Get the actual image being used
echo "🐳 Image Analysis for: $IMAGE"

# Check if we can exec into the pod
if kubectl exec -n "$NAMESPACE" "$POD" -- true 2>/dev/null; then
  echo "✅ Pod is ready for exec commands"
  
  # Comprehensive file system check
  echo -e "\n📁 File System Analysis:"
  
  # Check /app directory
  echo "📂 /app directory contents:"
  kubectl exec -n "$NAMESPACE" "$POD" -- ls -la /app/ 2>/dev/null || echo "❌ /app directory not found or inaccessible"
  
  # Check for binary in expected location
  if kubectl exec -n "$NAMESPACE" "$POD" -- test -f /app/sematic-cache 2>/dev/null; then
    echo "✅ Binary found at /app/sematic-cache"
    kubectl exec -n "$NAMESPACE" "$POD" -- ls -la /app/sematic-cache
    kubectl exec -n "$NAMESPACE" "$POD" -- file /app/sematic-cache 2>/dev/null || echo "❌ Cannot determine file type"
  else
    echo "❌ Binary NOT found at /app/sematic-cache"
  fi
  
  # Search for binary everywhere
  echo -e "\n🔍 Searching entire filesystem for 'sematic-cache':"
  kubectl exec -n "$NAMESPACE" "$POD" -- find / -name "*sematic*" -type f 2>/dev/null | head -20 || echo "❌ Search failed"
  
  # Check working directory
  echo -e "\n📍 Current working directory:"
  kubectl exec -n "$NAMESPACE" "$POD" -- pwd 2>/dev/null || echo "❌ Cannot get pwd"
  kubectl exec -n "$NAMESPACE" "$POD" -- ls -la 2>/dev/null || echo "❌ Cannot list current directory"
  
  # Check environment and PATH
  echo -e "\n🌍 Environment Analysis:"
  kubectl exec -n "$NAMESPACE" "$POD" -- env | grep -E "(PATH|HOME|PWD)" 2>/dev/null || echo "❌ Cannot get environment"
  
  # System info
  echo -e "\n🏗️ System Information:"
  kubectl exec -n "$NAMESPACE" "$POD" -- uname -a 2>/dev/null || echo "❌ Cannot get system info"
  kubectl exec -n "$NAMESPACE" "$POD" -- cat /etc/os-release 2>/dev/null | head -5 || echo "❌ Cannot get OS info"
  
else
  echo "⚠️ Pod is not ready for exec - analyzing image directly"
fi

# DOCKER IMAGE ANALYSIS (The most important part)
echo -e "\n🐳 DOCKER IMAGE DEEP ANALYSIS:"

if command -v docker >/dev/null 2>&1; then
  
  # Check if the exact image exists locally
  if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "✅ Image '$IMAGE' found locally"
    
    # Get image details
    echo -e "\n📋 Image Configuration:"
    docker image inspect "$IMAGE" | jq -r '.[0] | {
      "Created": .Created,
      "Architecture": .Architecture,
      "Os": .Os,
      "Size": .Size,
      "Cmd": .Config.Cmd,
      "Entrypoint": .Config.Entrypoint,
      "WorkingDir": .Config.WorkingDir,
      "Env": .Config.Env
    }' 2>/dev/null || echo "❌ Cannot parse image config"
    
    # Test run the image locally to see what happens
    echo -e "\n🧪 TESTING IMAGE LOCALLY:"
    echo "Running: docker run --rm '$IMAGE' ls -la /app/"
    
    if timeout 10s docker run --rm "$IMAGE" ls -la /app/ 2>&1; then
      echo "✅ Successfully listed /app/ directory"
    else
      echo "❌ Failed to list /app/ directory"
    fi
    
    echo -e "\nChecking if binary exists in image:"
    if timeout 10s docker run --rm "$IMAGE" test -f /app/sematic-cache 2>&1; then
      echo "✅ Binary exists in image"
      echo "Binary details:"
      timeout 10s docker run --rm "$IMAGE" ls -la /app/sematic-cache 2>&1 || echo "❌ Cannot get binary details"
      timeout 10s docker run --rm "$IMAGE" file /app/sematic-cache 2>&1 || echo "❌ Cannot determine file type"
    else
      echo "❌ Binary does NOT exist in image at /app/sematic-cache"
      
      echo -e "\nSearching for sematic-cache binary in image:"
      timeout 15s docker run --rm "$IMAGE" find / -name "*sematic*" -type f 2>/dev/null | head -10 || echo "❌ Search failed or timed out"
    fi
    
    # Try to run the exact command that Kubernetes is trying to run
    echo -e "\n🎯 TESTING EXACT K8S COMMAND:"
    echo "Running: docker run --rm '$IMAGE' /app/sematic-cache -address=:8080"
    
    if timeout 5s docker run --rm "$IMAGE" /app/sematic-cache -address=:8080 2>&1; then
      echo "✅ Command executed (may have failed for other reasons)"
    else
      echo "❌ Command failed - confirms the issue"
    fi
    
    # Check what's actually in the image at root and working directory
    echo -e "\n📂 Image Root Directory:"
    timeout 10s docker run --rm "$IMAGE" ls -la / 2>&1 | head -20 || echo "❌ Cannot list root"
    
    echo -e "\n📂 Image Working Directory:"
    timeout 10s docker run --rm "$IMAGE" pwd 2>&1 || echo "❌ Cannot get working directory"
    timeout 10s docker run --rm "$IMAGE" ls -la 2>&1 | head -20 || echo "❌ Cannot list working directory"
    
  else
    echo "❌ Image '$IMAGE' not found locally"
    
    # Check what images we do have
    echo -e "\n📋 Available local images related to sematic-cache:"
    docker images | grep -E "(sematic|cache)" || echo "No related images found"
    
    echo -e "\n📋 All local images:"
    docker images | head -10
  fi
  
  # Check for alternative image names/tags
  echo -e "\n🔍 Checking for similar images:"
  docker images | grep -i sematic || echo "No sematic images found"
  
else
  echo "❌ Docker not available for image analysis"
fi

# DOCKERFILE ANALYSIS
echo -e "\n📄 DOCKERFILE ANALYSIS:"

# Try to find and analyze the Dockerfile
DOCKERFILE_PATHS=(
  "deploy/docker/Dockerfile"
  "Dockerfile"
  "../docker/Dockerfile"
  "../../Dockerfile"
)

DOCKERFILE_FOUND=""
for dockerfile_path in "${DOCKERFILE_PATHS[@]}"; do
  if [[ -f "$dockerfile_path" ]]; then
    DOCKERFILE_FOUND="$dockerfile_path"
    break
  fi
done

if [[ -n "$DOCKERFILE_FOUND" ]]; then
  echo "✅ Found Dockerfile at: $DOCKERFILE_FOUND"
  echo -e "\n📋 Dockerfile contents:"
  cat "$DOCKERFILE_FOUND"
  
  echo -e "\n🔍 Key Dockerfile analysis:"
  echo "COPY/ADD commands:"
  grep -n -E "^(COPY|ADD)" "$DOCKERFILE_FOUND" || echo "No COPY/ADD commands found"
  
  echo -e "\nRUN commands:"
  grep -n -E "^RUN" "$DOCKERFILE_FOUND" || echo "No RUN commands found"
  
  echo -e "\nWORKDIR commands:"
  grep -n -E "^WORKDIR" "$DOCKERFILE_FOUND" || echo "No WORKDIR commands found"
  
  echo -e "\nCMD/ENTRYPOINT commands:"
  grep -n -E "^(CMD|ENTRYPOINT)" "$DOCKERFILE_FOUND" || echo "No CMD/ENTRYPOINT commands found"
  
else
  echo "❌ Dockerfile not found in common locations"
  echo "Searched in: ${DOCKERFILE_PATHS[*]}"
fi

# COMPREHENSIVE TROUBLESHOOTING
echo -e "\n💡 COMPREHENSIVE TROUBLESHOOTING GUIDE:"

if grep -q "no such file or directory" <(kubectl logs -n "$NAMESPACE" "$POD" 2>/dev/null || echo ""); then
  echo "🚨 CRITICAL ISSUE: Binary not found at /app/sematic-cache"
  echo ""
  echo "🔧 IMMEDIATE ACTIONS TO TAKE:"
  echo ""
  echo "1️⃣ VERIFY YOUR BUILD PROCESS:"
  echo "   cd $(git rev-parse --show-toplevel 2>/dev/null || pwd)"
  echo "   docker build -t sematic-cache:test -f deploy/docker/Dockerfile ."
  echo "   docker run --rm sematic-cache:test ls -la /app/"
  echo "   docker run --rm sematic-cache:test /app/sematic-cache --help"
  echo ""
  echo "2️⃣ CHECK YOUR DOCKERFILE:"
  echo "   - Ensure your binary is being built correctly"
  echo "   - Verify COPY/ADD commands are copying the binary to /app/sematic-cache"
  echo "   - Check that the binary has executable permissions"
  echo "   - Verify you're using the correct base image architecture"
  echo ""
  echo "3️⃣ VERIFY BINARY EXISTS BEFORE DOCKER BUILD:"
  echo "   ls -la \$(find . -name 'sematic-cache' -type f)"
  echo "   file \$(find . -name 'sematic-cache' -type f)"
  echo ""
  echo "4️⃣ COMMON DOCKERFILE ISSUES TO CHECK:"
  echo "   ❌ Missing: COPY ./sematic-cache /app/sematic-cache"
  echo "   ❌ Wrong path: COPY ./bin/sematic-cache /app/"
  echo "   ❌ No executable permission: RUN chmod +x /app/sematic-cache"
  echo "   ❌ Multi-stage build issues: binary not in final stage"
  echo "   ❌ Architecture mismatch: building for wrong platform"
  echo ""
  echo "5️⃣ DEBUG COMMANDS TO RUN:"
  echo "   # Check what's actually in your built image"
  echo "   docker run --rm -it sematic-cache:test sh"
  echo "   # In the container:"
  echo "   #   ls -la /app/"
  echo "   #   find / -name '*sematic*' 2>/dev/null"
  echo "   #   which sematic-cache"
fi

if [[ "$RESTART_COUNT" -gt 2 ]]; then
  echo ""
  echo "🔄 HIGH RESTART COUNT DETECTED ($RESTART_COUNT restarts)"
  echo "   This suggests the container starts but immediately crashes"
  echo "   Common causes:"
  echo "   - Binary doesn't exist (current issue)"
  echo "   - Binary wrong architecture (arm64 vs amd64)"
  echo "   - Missing dependencies in the container"
  echo "   - Database connection failures"
  echo "   - Permission issues"
fi

# Check if image tag matches what's expected
if [[ "$IMAGE" != "sematic-cache:latest" ]]; then
  echo ""
  echo "⚠️  IMAGE TAG MISMATCH DETECTED:"
  echo "   Expected: sematic-cache:latest"
  echo "   Actual:   $IMAGE"
  echo "   This suggests you may have built with a different tag"
  echo ""
  echo "   Solutions:"
  echo "   1. Rebuild with correct tag: docker build -t sematic-cache:latest ."
  echo "   2. Update your k8s manifest to use: $IMAGE"
  echo "   3. Re-import with: k3d image import $IMAGE -c sematic-cache"
fi

echo ""
echo "🎯 RECOMMENDED IMMEDIATE STEPS:"
echo "1. First, test your Docker image locally:"
echo "   docker run --rm sematic-cache:test ls -la /app/"
echo ""
echo "2. If binary is missing, check your build process:"
echo "   - Review your Dockerfile"
echo "   - Ensure binary is built before Docker build"
echo "   - Check COPY commands in Dockerfile"
echo ""
echo "3. Once fixed, rebuild and redeploy:"
echo "   deploy/k8s/cluster.sh down"
echo "   deploy/k8s/cluster.sh up"
echo "   deploy/k8s/dev.sh build"
echo "   deploy/k8s/dev.sh deploy"
echo ""
echo "4. If still failing, run this debug script again for deeper analysis"

echo -e "\n✅ DEEP ANALYSIS COMPLETE!"
echo ""
echo "📋 NEXT STEPS SUMMARY:"
echo "   1. Check Dockerfile and build process"
echo "   2. Test Docker image locally first"
echo "   3. Fix any issues found"
echo "   4. Rebuild and redeploy"
echo "   5. Re-run this debug script if needed"