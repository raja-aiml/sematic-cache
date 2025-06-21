package validation

import (
	"fmt"
	"os"
	"strings"

	"gopkg.in/yaml.v3"
)

// ManifestValidator validates Kubernetes manifest files
type ManifestValidator struct {
	// Add any configuration fields if needed
}

// NewManifestValidator creates a new manifest validator
func NewManifestValidator() *ManifestValidator {
	return &ManifestValidator{}
}

// ValidateFile validates a single manifest file
func (v *ManifestValidator) ValidateFile(filename string) (*ValidationResult, error) {
	result := NewValidationResult()

	// Read file
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	// Split by document separator (---)
	documents := strings.Split(string(data), "\n---\n")

	for i, doc := range documents {
		doc = strings.TrimSpace(doc)
		if doc == "" {
			continue
		}

		v.validateDocument(doc, i+1, filename, result)
	}

	return result, nil
}

// validateDocument validates a single YAML document
func (v *ManifestValidator) validateDocument(doc string, index int, filename string, result *ValidationResult) {
	var manifest map[string]interface{}

	// Parse YAML
	if err := yaml.Unmarshal([]byte(doc), &manifest); err != nil {
		result.AddError("Document %d: Invalid YAML - %v", index, err)
		return
	}

	// Check required fields
	if _, ok := manifest["apiVersion"]; !ok {
		result.AddError("Document %d: Missing apiVersion", index)
	}

	if _, ok := manifest["kind"]; !ok {
		result.AddError("Document %d: Missing kind", index)
		return
	}

	kind, _ := manifest["kind"].(string)
	apiVersion, _ := manifest["apiVersion"].(string)

	// Validate based on kind
	switch kind {
	case "Deployment":
		v.validateDeployment(manifest, index, result)
	case "Service":
		v.validateService(manifest, index, result)
	case "ConfigMap":
		v.validateConfigMap(manifest, index, result)
	case "Secret":
		v.validateSecret(manifest, index, result)
	case "PersistentVolumeClaim":
		v.validatePVC(manifest, index, result)
	case "NetworkPolicy":
		v.validateNetworkPolicy(manifest, index, result)
	case "Ingress":
		v.validateIngress(manifest, index, result)
	case "Job":
		v.validateJob(manifest, index, result)
	case "CronJob":
		v.validateCronJob(manifest, index, result)
	case "StatefulSet":
		v.validateStatefulSet(manifest, index, result)
	case "DaemonSet":
		v.validateDaemonSet(manifest, index, result)
	default:
		// Unknown kinds are not necessarily invalid
		result.AddInfo("Document %d: %s/%s", index, apiVersion, kind)
	}

	// Check metadata
	v.validateMetadata(manifest, index, result)
}

// validateMetadata validates common metadata fields
func (v *ManifestValidator) validateMetadata(manifest map[string]interface{}, index int, result *ValidationResult) {
	metadata, ok := manifest["metadata"].(map[string]interface{})
	if !ok {
		result.AddError("Document %d: Missing or invalid metadata", index)
		return
	}

	if _, ok := metadata["name"]; !ok {
		result.AddError("Document %d: Missing metadata.name", index)
	}

	// Validate name format
	if name, ok := metadata["name"].(string); ok {
		if !isValidKubernetesName(name) {
			result.AddError("Document %d: Invalid name '%s' - must be lowercase alphanumeric or '-'", index, name)
		}
	}

	// Check for recommended labels
	labels, hasLabels := metadata["labels"].(map[string]interface{})
	if !hasLabels || len(labels) == 0 {
		result.AddWarning("Document %d: No labels defined - consider adding labels for organization", index)
	} else {
		// Check for recommended labels
		recommendedLabels := []string{"app", "app.kubernetes.io/name", "app.kubernetes.io/component"}
		hasRecommended := false
		for _, label := range recommendedLabels {
			if _, ok := labels[label]; ok {
				hasRecommended = true
				break
			}
		}
		if !hasRecommended {
			result.AddWarning("Document %d: Consider using standard labels like 'app' or 'app.kubernetes.io/name'", index)
		}
	}
}

// validateDeployment validates Deployment specific fields
func (v *ManifestValidator) validateDeployment(manifest map[string]interface{}, index int, result *ValidationResult) {
	spec, ok := manifest["spec"].(map[string]interface{})
	if !ok {
		result.AddError("Document %d: Deployment missing spec", index)
		return
	}

	// Check replicas
	if replicas, ok := spec["replicas"].(int); ok {
		if replicas < 1 {
			result.AddWarning("Document %d: Deployment has %d replicas", index, replicas)
		}
	}

	// Check selector
	if _, ok := spec["selector"]; !ok {
		result.AddError("Document %d: Deployment missing spec.selector", index)
	}

	// Check template
	template, ok := spec["template"].(map[string]interface{})
	if !ok {
		result.AddError("Document %d: Deployment missing spec.template", index)
		return
	}

	// Validate pod template
	v.validatePodTemplate(template, index, "Deployment", result)
}

// validateService validates Service specific fields
func (v *ManifestValidator) validateService(manifest map[string]interface{}, index int, result *ValidationResult) {
	spec, ok := manifest["spec"].(map[string]interface{})
	if !ok {
		result.AddError("Document %d: Service missing spec", index)
		return
	}

	// Check ports
	ports, ok := spec["ports"].([]interface{})
	if !ok || len(ports) == 0 {
		result.AddError("Document %d: Service missing spec.ports", index)
		return
	}

	// Validate each port
	for i, port := range ports {
		if portMap, ok := port.(map[string]interface{}); ok {
			if _, ok := portMap["port"]; !ok {
				result.AddError("Document %d: Service port %d missing 'port' field", index, i)
			}

			// Check for targetPort when not using headless service
			if spec["clusterIP"] != "None" {
				if _, ok := portMap["targetPort"]; !ok {
					result.AddWarning("Document %d: Service port %d missing 'targetPort' - will default to 'port' value", index, i)
				}
			}
		}
	}

	// Check selector for non-headless services
	if spec["clusterIP"] != "None" && spec["type"] != "ExternalName" {
		if _, ok := spec["selector"]; !ok {
			result.AddWarning("Document %d: Service missing selector - will not route traffic to any pods", index)
		}
	}
}

// validatePodTemplate validates pod template specifications
func (v *ManifestValidator) validatePodTemplate(template map[string]interface{}, index int, kind string, result *ValidationResult) {
	// Check metadata
	if metadata, ok := template["metadata"].(map[string]interface{}); ok {
		if labels, ok := metadata["labels"].(map[string]interface{}); !ok || len(labels) == 0 {
			result.AddWarning("Document %d: %s pod template missing labels - required for service selection", index, kind)
		}
	}

	// Check spec
	spec, ok := template["spec"].(map[string]interface{})
	if !ok {
		result.AddError("Document %d: %s pod template missing spec", index, kind)
		return
	}

	// Check containers
	containers, ok := spec["containers"].([]interface{})
	if !ok || len(containers) == 0 {
		result.AddError("Document %d: %s pod template missing containers", index, kind)
		return
	}

	// Validate each container
	for i, container := range containers {
		if containerMap, ok := container.(map[string]interface{}); ok {
			v.validateContainer(containerMap, index, i, kind, result)
		}
	}
}

// validateContainer validates container specifications
func (v *ManifestValidator) validateContainer(container map[string]interface{}, docIndex, containerIndex int, kind string, result *ValidationResult) {
	// Check name
	if _, ok := container["name"]; !ok {
		result.AddError("Document %d: %s container %d missing name", docIndex, kind, containerIndex)
	}

	// Check image
	if image, ok := container["image"].(string); !ok {
		result.AddError("Document %d: %s container %d missing image", docIndex, kind, containerIndex)
	} else if image == "" {
		result.AddError("Document %d: %s container %d has empty image", docIndex, kind, containerIndex)
	} else if !strings.Contains(image, ":") {
		result.AddWarning("Document %d: %s container %d image '%s' missing tag - will use 'latest'", docIndex, kind, containerIndex, image)
	}

	// Check resources
	if _, ok := container["resources"]; !ok {
		result.AddWarning("Document %d: %s container %d missing resource limits/requests", docIndex, kind, containerIndex)
	}

	// Check security context
	if _, ok := container["securityContext"]; !ok {
		result.AddInfo("Document %d: %s container %d missing securityContext - consider adding for security", docIndex, kind, containerIndex)
	}
}

// Additional validators for other resource types

func (v *ManifestValidator) validateConfigMap(manifest map[string]interface{}, index int, result *ValidationResult) {
	// ConfigMaps should have either data or binaryData
	hasData := false
	if data, ok := manifest["data"]; ok && data != nil {
		hasData = true
	}
	if binaryData, ok := manifest["binaryData"]; ok && binaryData != nil {
		hasData = true
	}

	if !hasData {
		result.AddWarning("Document %d: ConfigMap has no data or binaryData", index)
	}
}

func (v *ManifestValidator) validateSecret(manifest map[string]interface{}, index int, result *ValidationResult) {
	// Check type
	if _, ok := manifest["type"]; !ok {
		result.AddInfo("Document %d: Secret missing type - will default to 'Opaque'", index)
	}

	// Secrets should have either data or stringData
	hasData := false
	if data, ok := manifest["data"]; ok && data != nil {
		hasData = true
	}
	if stringData, ok := manifest["stringData"]; ok && stringData != nil {
		hasData = true
	}

	if !hasData {
		result.AddWarning("Document %d: Secret has no data or stringData", index)
	}
}

func (v *ManifestValidator) validatePVC(manifest map[string]interface{}, index int, result *ValidationResult) {
	spec, ok := manifest["spec"].(map[string]interface{})
	if !ok {
		result.AddError("Document %d: PVC missing spec", index)
		return
	}

	// Check access modes
	if _, ok := spec["accessModes"]; !ok {
		result.AddError("Document %d: PVC missing spec.accessModes", index)
	}

	// Check resources
	if resources, ok := spec["resources"].(map[string]interface{}); !ok {
		result.AddError("Document %d: PVC missing spec.resources", index)
	} else {
		if requests, ok := resources["requests"].(map[string]interface{}); !ok {
			result.AddError("Document %d: PVC missing spec.resources.requests", index)
		} else {
			if _, ok := requests["storage"]; !ok {
				result.AddError("Document %d: PVC missing spec.resources.requests.storage", index)
			}
		}
	}
}

func (v *ManifestValidator) validateNetworkPolicy(manifest map[string]interface{}, index int, result *ValidationResult) {
	spec, ok := manifest["spec"].(map[string]interface{})
	if !ok {
		result.AddError("Document %d: NetworkPolicy missing spec", index)
		return
	}

	// Check podSelector
	if _, ok := spec["podSelector"]; !ok {
		result.AddError("Document %d: NetworkPolicy missing spec.podSelector", index)
	}

	// Check if it has any rules
	hasIngress := false
	hasEgress := false
	if _, ok := spec["ingress"]; ok {
		hasIngress = true
	}
	if _, ok := spec["egress"]; ok {
		hasEgress = true
	}

	if !hasIngress && !hasEgress {
		result.AddWarning("Document %d: NetworkPolicy has no ingress or egress rules", index)
	}
}

func (v *ManifestValidator) validateIngress(manifest map[string]interface{}, index int, result *ValidationResult) {
	spec, ok := manifest["spec"].(map[string]interface{})
	if !ok {
		result.AddError("Document %d: Ingress missing spec", index)
		return
	}

	// Check rules
	if rules, ok := spec["rules"].([]interface{}); !ok || len(rules) == 0 {
		result.AddWarning("Document %d: Ingress has no rules", index)
	}
}

func (v *ManifestValidator) validateJob(manifest map[string]interface{}, index int, result *ValidationResult) {
	spec, ok := manifest["spec"].(map[string]interface{})
	if !ok {
		result.AddError("Document %d: Job missing spec", index)
		return
	}

	// Check template
	if template, ok := spec["template"].(map[string]interface{}); !ok {
		result.AddError("Document %d: Job missing spec.template", index)
	} else {
		v.validatePodTemplate(template, index, "Job", result)

		// Jobs should have restartPolicy Never or OnFailure
		if podSpec, ok := template["spec"].(map[string]interface{}); ok {
			if restartPolicy, ok := podSpec["restartPolicy"].(string); ok {
				if restartPolicy != "Never" && restartPolicy != "OnFailure" {
					result.AddError("Document %d: Job restartPolicy must be 'Never' or 'OnFailure', got '%s'", index, restartPolicy)
				}
			} else {
				result.AddError("Document %d: Job missing restartPolicy", index)
			}
		}
	}
}

func (v *ManifestValidator) validateCronJob(manifest map[string]interface{}, index int, result *ValidationResult) {
	spec, ok := manifest["spec"].(map[string]interface{})
	if !ok {
		result.AddError("Document %d: CronJob missing spec", index)
		return
	}

	// Check schedule
	if _, ok := spec["schedule"]; !ok {
		result.AddError("Document %d: CronJob missing spec.schedule", index)
	}

	// Check jobTemplate
	if jobTemplate, ok := spec["jobTemplate"].(map[string]interface{}); !ok {
		result.AddError("Document %d: CronJob missing spec.jobTemplate", index)
	} else {
		// Validate job template
		if jobSpec, ok := jobTemplate["spec"].(map[string]interface{}); ok {
			fakeJob := map[string]interface{}{
				"spec": jobSpec,
			}
			v.validateJob(fakeJob, index, result)
		}
	}
}

func (v *ManifestValidator) validateStatefulSet(manifest map[string]interface{}, index int, result *ValidationResult) {
	spec, ok := manifest["spec"].(map[string]interface{})
	if !ok {
		result.AddError("Document %d: StatefulSet missing spec", index)
		return
	}

	// Check serviceName
	if _, ok := spec["serviceName"]; !ok {
		result.AddError("Document %d: StatefulSet missing spec.serviceName", index)
	}

	// Check selector
	if _, ok := spec["selector"]; !ok {
		result.AddError("Document %d: StatefulSet missing spec.selector", index)
	}

	// Check template
	if template, ok := spec["template"].(map[string]interface{}); !ok {
		result.AddError("Document %d: StatefulSet missing spec.template", index)
	} else {
		v.validatePodTemplate(template, index, "StatefulSet", result)
	}
}

func (v *ManifestValidator) validateDaemonSet(manifest map[string]interface{}, index int, result *ValidationResult) {
	spec, ok := manifest["spec"].(map[string]interface{})
	if !ok {
		result.AddError("Document %d: DaemonSet missing spec", index)
		return
	}

	// Check selector
	if _, ok := spec["selector"]; !ok {
		result.AddError("Document %d: DaemonSet missing spec.selector", index)
	}

	// Check template
	if template, ok := spec["template"].(map[string]interface{}); !ok {
		result.AddError("Document %d: DaemonSet missing spec.template", index)
	} else {
		v.validatePodTemplate(template, index, "DaemonSet", result)
	}
}

// Helper functions

func isValidKubernetesName(name string) bool {
	if len(name) == 0 || len(name) > 253 {
		return false
	}

	for i, ch := range name {
		if !((ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '-' || ch == '.') {
			return false
		}

		// Cannot start or end with dash or dot
		if (i == 0 || i == len(name)-1) && (ch == '-' || ch == '.') {
			return false
		}
	}

	return true
}
