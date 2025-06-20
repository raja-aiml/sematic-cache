package constants

import "time"

const (
	// Cluster defaults
	DefaultClusterName = "local-k8s"
	DefaultAPIPort     = "6550"

	// Container defaults
	DefaultImageName = "app:local"

	// Namespace defaults
	AppNamespace   = "app"
	InfraNamespace = "infra"

	// Timeout defaults
	DefaultTimeout        = 5 * time.Minute
	DefaultBuildTimeout   = 10 * time.Minute
	DefaultHTTPTimeout    = 5 * time.Second
	DefaultClusterTimeout = 300 * time.Second

	// Port mappings
	HTTPPort  = "8080:80"
	HTTPSPort = "8443:443"

	// Secret names
	AppSecretName = "app-secrets"

	// Default database URL for local deployment
	DefaultDatabaseURL = "postgres://postgres:postgres@postgres.infra.svc.cluster.local:5432/appdb?sslmode=disable"

	// Blueprint path defaults
	DefaultBlueprintPath = "iaac/blueprint"
	DefaultInfraPath     = "infra"
	DefaultAppPath       = "app"
)
