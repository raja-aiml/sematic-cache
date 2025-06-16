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
echo -e "\n🔍 Advanced Debugging:"

# Check if we can exec into the pod
if kubectl exec -n "$NAMESPACE" "$POD" -- true 2>/dev/null; then
  echo "✅ Pod is ready for exec commands"
  
  # Check if binary exists
  echo "📁 Checking for sematic-cache binary:"
  if kubectl exec -n "$NAMESPACE" "$POD" -- test -f /app/sematic-cache 2>/dev/null; then
    echo "✅ Binary exists at /app/sematic-cache"
    
    # Check binary properties
    echo "🔍 Binary properties:"
    kubectl exec -n "$NAMESPACE" "$POD" -- ls -la /app/sematic-cache 2>/dev/null || echo "❌ Cannot list binary properties"
    kubectl exec -n "$NAMESPACE" "$POD" -- file /app/sematic-cache 2>/dev/null || echo "❌ Cannot inspect binary format"
    
    # Check if it's executable
    if kubectl exec -n "$NAMESPACE" "$POD" -- test -x /app/sematic-cache 2>/dev/null; then
      echo "✅ Binary is executable"
    else
      echo "❌ Binary is not executable"
    fi
  else
    echo "❌ Binary not found at /app/sematic-cache"
    
    # Check if it exists elsewhere
    echo "🔍 Searching for sematic-cache binary in common locations:"
    for path in "/usr/local/bin/sematic-cache" "/usr/bin/sematic-cache" "/bin/sematic-cache" "/sematic-cache"; do
      if kubectl exec -n "$NAMESPACE" "$POD" -- test -f "$path" 2>/dev/null; then
        echo "✅ Found binary at: $path"
      fi
    done
    
    # List /app directory contents
    echo "📁 Contents of /app directory:"
    kubectl exec -n "$NAMESPACE" "$POD" -- ls -la /app/ 2>/dev/null || echo "❌ Cannot list /app directory"
  fi
  
  # Check system architecture
  echo "🏗️ Container architecture:"
  kubectl exec -n "$NAMESPACE" "$POD" -- uname -a 2>/dev/null || echo "❌ Cannot get system info"
  
else
  echo "⚠️ Pod is not ready for exec (likely due to crashes)"
  
  # Try to inspect the image directly if possible
  echo "🐳 Checking Docker image locally..."
  if command -v docker >/dev/null 2>&1; then
    if docker image inspect "$IMAGE" >/dev/null 2>&1; then
      echo "✅ Image '$IMAGE' found locally"
      echo "🔍 Image layers and commands:"
      docker image inspect "$IMAGE" | jq -r '.[0].Config.Cmd // []' 2>/dev/null || echo "❌ Cannot inspect image config"
    else
      echo "❌ Image '$IMAGE' not found locally"
    fi
  else
    echo "❌ Docker not available for image inspection"
  fi
fi

# Suggest fixes based on common issues
echo -e "\n💡 Troubleshooting Suggestions:"

if grep -q "no such file or directory" <(kubectl logs -n "$NAMESPACE" "$POD" 2>/dev/null || echo ""); then
  echo "🔧 Issue: Binary not found"
  echo "   Solutions:"
  echo "   1. Check if the Docker build completed successfully"
  echo "   2. Verify the binary path in the Dockerfile"
  echo "   3. Ensure the binary is copied to /app/sematic-cache in the image"
  echo "   4. Check if the binary was built for the correct architecture"
fi

if [[ "$RESTART_COUNT" -gt 0 ]]; then
  echo "🔧 High restart count detected"
  echo "   Solutions:"
  echo "   1. Check application startup requirements (database connectivity, etc.)"
  echo "   2. Verify environment variables are set correctly"
  echo "   3. Check resource limits and requests"
fi

echo -e "\n✅ Debug analysis complete!"
echo "💡 To rebuild and redeploy:"
echo "   deploy/k8s/dev.sh build"
echo "   deploy/k8s/dev.sh deploy"