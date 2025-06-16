#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="app"
APP_LABEL="app=sematic-cache"
IMAGE_NAME="sematic-cache:latest"

echo "🔍 Searching for pods with label '$APP_LABEL' in namespace '$NAMESPACE'..."

POD=$(kubectl get pods -n "$NAMESPACE" -l "$APP_LABEL" -o json \
  | jq -r '.items | sort_by(-.status.containerStatuses[0].restartCount) | .[0].metadata.name')

if [[ -z "${POD:-}" || "$POD" == "null" ]]; then
  echo "❌ No pods found with label '$APP_LABEL'"
  exit 1
fi

echo "🎯 Selected pod: $POD"

echo -e "\n📦 Pods in '$NAMESPACE' namespace:"
kubectl get pods -n "$NAMESPACE"

# Get container details
POD_JSON=$(kubectl get pod -n "$NAMESPACE" "$POD" -o json)
CONTAINER=$(echo "$POD_JSON" | jq -r '.spec.containers[0].name')
IMAGE=$(echo "$POD_JSON" | jq -r '.spec.containers[0].image')
RESTARTS=$(echo "$POD_JSON" | jq -r '.status.containerStatuses[0].restartCount')
STATE=$(echo "$POD_JSON" | jq -r '.status.containerStatuses[0].state | keys[0]')
EXIT_CODE=$(echo "$POD_JSON" | jq -r '.status.containerStatuses[0].state.terminated.exitCode // "N/A"')
REASON=$(echo "$POD_JSON" | jq -r '.status.containerStatuses[0].state.terminated.reason // "N/A"')
LAST_EXIT=$(echo "$POD_JSON" | jq -r '.status.containerStatuses[0].lastState.terminated.exitCode // "N/A"')
LAST_REASON=$(echo "$POD_JSON" | jq -r '.status.containerStatuses[0].lastState.terminated.reason // "N/A"')
PHASE=$(echo "$POD_JSON" | jq -r '.status.phase')
READY=$(echo "$POD_JSON" | jq -r '.status.containerStatuses[0].ready')

echo -e "\n📊 Pod Status Summary:"
echo "🆔 Pod Name:       $POD"
echo "📊 Phase:          $PHASE"
echo "✅ Ready:          $READY"
echo "🔁 Restart Count:  $RESTARTS"
echo "🐳 Image:          $IMAGE"
echo "📌 Current State:  $STATE"
echo "💥 Exit Code:      $EXIT_CODE"
echo "📖 Reason:         $REASON"

echo -e "\n🔄 Last Termination:"
echo "💥 Last Exit Code: $LAST_EXIT"
echo "📖 Last Reason:    $LAST_REASON"

echo -e "\n📅 Recent Events:"
kubectl get events -n "$NAMESPACE" --field-selector involvedObject.name="$POD" --sort-by='.lastTimestamp' | tail -n 10

echo -e "\n📜 Container Logs:"
if kubectl logs -n "$NAMESPACE" "$POD" > /tmp/sematic-cache.log 2>/dev/null; then
  tail -n 20 /tmp/sematic-cache.log
  echo "✅ Logs retrieved successfully"
else
  echo "⚠️ Failed to retrieve logs"
fi

echo -e "\n🔍 Advanced Debugging:"
if ! kubectl exec -n "$NAMESPACE" "$POD" -- true &>/dev/null; then
  echo "⚠️ Pod is not ready for exec (likely due to crashes)"
else
  echo "📁 /app/sematic-cache:"
  kubectl exec -n "$NAMESPACE" "$POD" -- ls -l /app/sematic-cache || echo "❌ Not found"
fi

echo "🐳 Checking Docker image locally..."
if docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
  echo "✅ Image '$IMAGE_NAME' found locally"
  echo "🔍 Image layers and commands:"
  docker history "$IMAGE_NAME" --no-trunc | head -n 5
else
  echo "❌ Image '$IMAGE_NAME' not found locally"
fi

echo -e "\n💡 Troubleshooting Suggestions:"
if [[ "$REASON" == "Error" && "$EXIT_CODE" == "255" ]]; then
  echo "🔧 Issue: Binary not found"
  echo "   Solutions:"
  echo "   1. Check if the Docker build completed successfully"
  echo "   2. Verify the binary path in the Dockerfile"
  echo "   3. Ensure the binary is copied to /app/sematic-cache in the image"
  echo "   4. Check if the binary was built for the correct architecture"
fi

if [[ "$RESTARTS" -gt 2 ]]; then
  echo "🔧 High restart count detected"
  echo "   Solutions:"
  echo "   1. Check application startup requirements (e.g., DB connection)"
  echo "   2. Verify required env vars"
  echo "   3. Inspect resource limits or crashes"
fi

echo -e "\n✅ Debug analysis complete!"
echo "💡 To rebuild and redeploy:"
echo "   deploy/k8s/dev.sh build"
echo "   deploy/k8s/dev.sh deploy"