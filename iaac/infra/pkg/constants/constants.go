// Package constants provides constants for the iaac infrastructure deployment.
package constants

const (
	// K3d specific defaults
	DefaultK3dRegistry = "k3d-local-registry:5000"
	DefaultK3sImage    = "rancher/k3s:v1.28.5-k3s1"

	// Additional namespace constants (core ones are in defaults.go)
	MonitoringNamespace = "monitoring"
	IstioNamespace      = "istio-system"
	LoggingNamespace    = "logging"
	TracingNamespace    = "tracing"

	// Component labels
	LabelApp       = "app"
	LabelComponent = "component"
	LabelScenario  = "scenario"
	LabelManaged   = "managed-by"
	ManagedByValue = "iaac"

	// Additional timeouts (core ones are in defaults.go)
	ResourceReadyTimeout   = 600 // seconds
	ClusterCreationTimeout = 300 // seconds
	DeploymentReadyTimeout = 600 // seconds
)
