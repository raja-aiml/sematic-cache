#!/bin/bash
set -euo pipefail

NAMESPACE="app"
APP_LABEL="app=sematic-cache"

echo "🔍 Searching for the most recently restarted pod with label '$APP_LABEL' in namespace '$NAMESPACE'..."

# Select pod with highest restart count
POD=$(kubectl get pods -n "$NAMESPACE" -l "$APP_LABEL" -o json \
  | jq -r '.items | sort_by(-.status.containerStatuses[0].restartCount) | .[0].metadata.name')

if [[ -z "${POD:-}" || "$POD" == "null" ]]; then
  echo "❌ No sematic-cache pods found in namespace $NAMESPACE"
  kubectl get pods -n "$NAMESPACE"
  exit 1
fi

echo -e "\n📦 Pods in '$NAMESPACE' namespace:"
kubectl get pods -n "$NAMESPACE"

echo -e "\n📘 Describing pod: $POD"
kubectl describe pod -n "$NAMESPACE" "$POD" || echo "⚠️ Failed to describe pod: $POD"

echo -e "\n📜 Logs from pod: $POD"
kubectl logs -n "$NAMESPACE" "$POD" || echo "⚠️ Could not fetch logs (pod may have restarted too recently)"

echo -e "\n🔍 Inspecting filesystem inside pod (if available)..."
if kubectl exec -n "$NAMESPACE" "$POD" -- true 2>/dev/null; then
  echo "📁 Checking /app/sematic-cache inside container:"
  kubectl exec -n "$NAMESPACE" "$POD" -- ls -l /app/sematic-cache || echo "❌ Binary not found"
  echo
  kubectl exec -n "$NAMESPACE" "$POD" -- file /app/sematic-cache || echo "❌ Cannot inspect binary format"
else
  echo "⚠️ Pod '$POD' is not ready for exec. It may be restarting too frequently."
fi