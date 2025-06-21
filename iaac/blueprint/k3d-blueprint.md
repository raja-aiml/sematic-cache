# K3D Blueprint - Complete Observability Development Stack

**Professional K3D blueprint with Istio service mesh and comprehensive observability**  
*PostgreSQL + Redis + Istio + Prometheus + Grafana + Loki + OpenTelemetry + Fluent Bit + OTEL Visualizer*

```
k3d-blueprint/
├── .gitignore                    # Patterns to exclude from version control
├── .gitattributes               # Git attribute rules (e.g., line endings, diffs)
├── .env.example                 # Template for environment variables
├── README.md                    # Project overview, quick start, and usage
├── Taskfile.yaml                # Task runner definitions (task CLI)
│
├── infra/                       # 🏗️ Kustomize-based Kubernetes manifests
│   ├── base/                    # Environment-agnostic base stack
│   │   ├── kustomization.yaml   # Includes all core base components
│   │   ├── namespaces.yaml      # Namespace definitions
│   │   ├── network-policies.yaml# Default security isolation
│   │   ├── resource-quotas.yaml # Resource governance
│   │   │
│   │   ├── postgres/            # PostgreSQL with pgvector
│   │   │   ├── kustomization.yaml
│   │   │   ├── configmap.yaml   # PostgreSQL configuration
│   │   │   ├── secret.yaml      # Database credentials
│   │   │   ├── deployment.yaml  # PostgreSQL deployment
│   │   │   ├── service.yaml     # Database service
│   │   │   ├── pvc.yaml         # Persistent volume claim
│   │   │   ├── pdb.yaml         # Pod disruption budget
│   │   │   ├── hpa.yaml         # Horizontal pod autoscaler
│   │   │   ├── network-policy.yaml # Network isolation
│   │   │   └── service-monitor.yaml # Prometheus monitoring
│   │   │
│   │   └── redis/               # Redis service stack
│   │       ├── kustomization.yaml
│   │       ├── configmap.yaml   # Redis configuration
│   │       ├── deployment.yaml  # Redis deployment
│   │       ├── service.yaml     # Redis service
│   │       ├── pvc.yaml         # Persistent storage
│   │       ├── pdb.yaml         # Pod disruption budget
│   │       ├── hpa.yaml         # Horizontal pod autoscaler
│   │       ├── network-policy.yaml # Network isolation
│   │       └── service-monitor.yaml # Prometheus monitoring
│
│   ├── overlays/                # 🔀 Environment-specific configurations
│   │   ├── local/               # Minimal local development config
│   │   │   ├── kustomization.yaml
│   │   │   ├── resource-limits.yaml
│   │   │   ├── storage-patch.yaml
│   │   │   └── istio-minimal.yaml
│   │   │
│   │   └── dev/                 # Full dev overlay with all modules
│   │       ├── kustomization.yaml
│   │       ├── namespace-patch.yaml
│   │       ├── postgres-patch.yaml
│   │       ├── redis-patch.yaml
│   │       ├── ingress-patch.yaml
│   │       ├── monitoring-patch.yaml
│   │       ├── istio-patch.yaml
│   │       └── debug-tools.yaml
│
│   └── modules/                 # 🧩 Optional add-on components
│       │
│       ├── istio/               # 🕸️ Istio Service Mesh
│       │   ├── kustomization.yaml
│       │   ├── base/            # Core Istio components
│       │   │   ├── kustomization.yaml
│       │   │   ├── namespace.yaml
│       │   │   ├── istio-operator.yaml
│       │   │   ├── control-plane.yaml
│       │   │   ├── gateway.yaml
│       │   │   ├── peer-authentication.yaml
│       │   │   └── telemetry.yaml
│       │   ├── security/        # Security policies
│       │   │   ├── kustomization.yaml
│       │   │   ├── authorization-policies.yaml
│       │   │   ├── request-authentication.yaml
│       │   │   └── mtls-policies.yaml
│       │   └── addons/          # Istio observability add-ons
│       │       ├── kustomization.yaml
│       │       └── prometheus-istio.yaml
│       │
│       ├── observability/       # 📊 Unified monitoring/logging/tracing
│       │   ├── kustomization.yaml
│       │   │
│       │   ├── prometheus/      # Prometheus + Alertmanager + rules
│       │   │   ├── kustomization.yaml
│       │   │   ├── namespace.yaml
│       │   │   ├── prometheus.yaml
│       │   │   ├── prometheus-config.yaml
│       │   │   ├── alertmanager.yaml
│       │   │   ├── service-monitor.yaml
│       │   │   ├── prometheus-rules.yaml
│       │   │   ├── rbac.yaml
│       │   │   └── storage.yaml
│       │   │
│       │   ├── grafana/         # Dashboards, datasources, PVCs
│       │   │   ├── kustomization.yaml
│       │   │   ├── namespace.yaml
│       │   │   ├── deployment.yaml
│       │   │   ├── service.yaml
│       │   │   ├── configmap.yaml
│       │   │   ├── secret.yaml
│       │   │   ├── pvc.yaml
│       │   │   ├── rbac.yaml
│       │   │   ├── ingress.yaml
│       │   │   ├── datasources.yaml
│       │   │   ├── dashboard-provider.yaml
│       │   │   └── dashboards/
│       │   │       ├── istio-mesh.json
│       │   │       ├── istio-service.json
│       │   │       ├── istio-workload.json
│       │   │       ├── istio-control-plane.json
│       │   │       ├── kubernetes-cluster.json
│       │   │       ├── prometheus-overview.json
│       │   │       ├── postgres-dashboard.json
│       │   │       ├── redis-dashboard.json
│       │   │       ├── nginx-ingress.json
│       │   │       ├── otel-traces.json
│       │   │       └── application-metrics.json
│       │   │
│       │   ├── logging/         # Loki + FluentBit stack
│       │   │   ├── kustomization.yaml
│       │   │   ├── namespace.yaml
│       │   │   ├── loki/
│       │   │   │   ├── kustomization.yaml
│       │   │   │   ├── deployment.yaml
│       │   │   │   ├── service.yaml
│       │   │   │   ├── configmap.yaml
│       │   │   │   ├── pvc.yaml
│       │   │   │   └── rbac.yaml
│       │   │   ├── fluent-bit/
│       │   │   │   ├── kustomization.yaml
│       │   │   │   ├── daemonset.yaml
│       │   │   │   ├── configmap.yaml
│       │   │   │   ├── rbac.yaml
│       │   │   │   └── service.yaml
│       │   │   └── promtail/
│       │   │       ├── kustomization.yaml
│       │   │       ├── daemonset.yaml
│       │   │       ├── configmap.yaml
│       │   │       └── rbac.yaml
│       │   │
│       │   └── tracing/         # OpenTelemetry + Tempo
│       │       ├── kustomization.yaml
│       │       ├── namespace.yaml
│       │       ├── opentelemetry/
│       │       │   ├── kustomization.yaml
│       │       │   ├── operator.yaml
│       │       │   ├── collector.yaml
│       │       │   ├── instrumentation.yaml
│       │       │   └── rbac.yaml
│       │       ├── tempo/
│       │       │   ├── kustomization.yaml
│       │       │   ├── deployment.yaml
│       │       │   ├── service.yaml
│       │       │   ├── configmap.yaml
│       │       │   └── pvc.yaml
│       │       └── otel-visualizer/
│       │           ├── kustomization.yaml
│       │           ├── deployment.yaml
│       │           ├── service.yaml
│       │           ├── configmap.yaml
│       │           └── ingress.yaml
│       │
│       │
│       ├── security/            # 🔒 PodSecurity, NetworkPolicy, RBAC, etc.
│       │   ├── kustomization.yaml
│       │   ├── pod-security-policies.yaml
│       │   ├── network-policies.yaml
│       │   ├── rbac.yaml
│       │   └── security-contexts.yaml
│       │
│
├── scenarios/                   # 🌐 Real-world blueprint compositions
│   ├── minimal/                 # Just the essentials (postgres + redis)
│   │   ├── kustomization.yaml
│   │   ├── README.md
│   │   └── values.yaml
│   │
│   ├── development/             # Full dev experience with observability
│   │   ├── kustomization.yaml
│   │   ├── README.md
│   │   ├── values.yaml
│   │   └── dev-tools.yaml
│   │
│   ├── service-mesh/            # Istio-focused deployment
│   │   ├── kustomization.yaml
│   │   ├── README.md
│   │   ├── values.yaml
│   │   └── mesh-config.yaml
│   │
│   ├── monitoring-only/         # Only Prometheus, Grafana, Loki, OTEL
│   │   ├── kustomization.yaml
│   │   ├── README.md
│   │   └── values.yaml
│   │
│   └── full-stack/              # Complete stack with everything enabled
│       ├── kustomization.yaml
│       ├── README.md
│       ├── values.yaml
│       └── production-config.yaml
│
│   ├── load-generator/          # Traffic generator for testing
│   │   ├── kustomization.yaml
│   │   ├── deployment.yaml
│   │   ├── configmap.yaml
│   │   └── service.yaml
│   │
│   └── test-client/             # Debugging and testing tools
│       ├── kustomization.yaml
│       ├── postgres-client.yaml
│       ├── redis-client.yaml
│       ├── curl-pod.yaml
│       └── debug-pod.yaml
│
├── scripts/                     # ⚙️ Automation and ops tools
│   ├── README.md                # Script catalog and usage guide
│   │
│   ├── lib/                     # Shared shell utilities
│   │   ├── common.sh            # General helper functions
│   │   ├── k8s.sh               # Kubectl/cluster helpers
│   │   ├── logging.sh           # Logging format helpers
│   │   ├── validation.sh        # Testing helper functions
│   │   ├── istio.sh             # Istio-specific helpers
│   │   └── monitoring.sh        # Monitoring setup helpers
│   │
│   ├── cluster/                 # Cluster lifecycle management
│   │   ├── cluster.sh           # Create k3d cluster
│   │
│   └── dev.sh                   # Main development script
│
├── validation-kit/              # ✅ Validation, testing, smoke checks
│   ├── README.md                # Test runner and testing philosophy
│   │
│   ├── tests/                   # Test definitions
│   │   ├── integration/
│   │   │   ├── postgres-test.yaml
│   │   │   ├── redis-test.yaml
│   │   │   ├── ingress-test.yaml
│   │   │   ├── istio-test.yaml
│   │   │   ├── monitoring-test.yaml
│   │   │   └── tracing-test.yaml
│   │   ├── performance/
│   │   │   ├── load-test.yaml
│   │   │   ├── stress-test.yaml
│   │   │   └── benchmark.yaml
│   │   ├── security/
│   │   │   ├── network-policy-test.yaml
│   │   │   ├── rbac-test.yaml
│   │   │   └── mtls-test.yaml
│   │   └── observability/
│   │       ├── metrics-test.yaml
│   │       ├── logs-test.yaml
│   │       ├── traces-test.yaml
│   │       └── alerts-test.yaml
│   │
│   ├── client-connections/      # Test pods to verify service access
│   │   ├── postgres-client.yaml
│   │   ├── redis-client.yaml
│   │   ├── curl-pod.yaml
│   │   ├── debug-pod.yaml
│   │   └── istio-proxy-test.yaml
│   │
│   ├── seed-data/               # Sample data for testing
│   │   ├── postgres/
│   │   │   ├── schema.sql
│   │   │   ├── sample-data.sql
│   │   │   └── test-queries.sql
│   │   ├── redis/
│   │   │   ├── sample-keys.redis
│   │   │   └── test-commands.txt
│   │   ├── monitoring/
│   │   │   ├── sample-metrics.json
│   │   │   ├── test-alerts.yaml
│   │   │   └── trace-examples.json
│   │   └── istio/
│   │       ├── traffic-scenarios.yaml
│   │       └── policy-tests.yaml
│   │
│   └── scripts/                 # Test automation
│       ├── run-tests.sh         # Main test runner
│       ├── smoke-test.sh        # Quick smoke tests
│       ├── integration-test.sh  # Full integration tests
│       ├── monitoring-test.sh   # Observability validation
│       ├── istio-test.sh        # Service mesh validation
│       ├── performance-test.sh  # Performance testing
│       ├── cleanup.sh           # Test cleanup
│       └── report.sh            # Generate test reports
│
└── hack/                        # 🔧 Dev and maintenance scripts
    ├── update-dependencies.sh   # Update project dependencies
    ├── generate-manifests.sh    # Render manifests from Kustomize
    ├── lint-all.sh              # Lint all manifests/scripts
    ├── test-all.sh              # Run all tests
    ├── verify-blueprint.sh      # Validate overall project structure
    ├── benchmark.sh             # Performance benchmarking
    └── release.sh               # Prepare releases
```
