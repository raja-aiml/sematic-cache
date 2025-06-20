package kubernetes

import (
	"bytes"
	"context"
	"fmt"
	"io"

	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer/yaml"
	yamlutil "k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"
	"path/filepath"
	"sigs.k8s.io/kustomize/api/krusty"
	"sigs.k8s.io/kustomize/kyaml/filesys"

	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/utils"
)

// ApplyConfig holds configuration for applying resources
type ApplyConfig struct {
	Config    *rest.Config
	Clientset *kubernetes.Clientset
	Dynamic   dynamic.Interface
	Mapper    meta.RESTMapper
	Logger    *utils.Logger
}

// NewApplyConfig creates a new apply configuration
func NewApplyConfig(kubeconfigPath string) (*ApplyConfig, error) {
	if kubeconfigPath == "" {
		if home := homedir.HomeDir(); home != "" {
			kubeconfigPath = filepath.Join(home, ".kube", "config")
		}
	}

	config, err := clientcmd.BuildConfigFromFlags("", kubeconfigPath)
	if err != nil {
		return nil, fmt.Errorf("failed to build config: %w", err)
	}

	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create clientset: %w", err)
	}

	dynamic, err := dynamic.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create dynamic client: %w", err)
	}

	gr, err := restmapper.GetAPIGroupResources(clientset.Discovery())
	if err != nil {
		return nil, fmt.Errorf("failed to get API group resources: %w", err)
	}

	mapper := restmapper.NewDiscoveryRESTMapper(gr)

	return &ApplyConfig{
		Config:    config,
		Clientset: clientset,
		Dynamic:   dynamic,
		Mapper:    mapper,
		Logger:    utils.NewLogger("kubernetes"),
	}, nil
}

// ApplyKustomizeSDK applies kustomize resources using the SDK
func (ac *ApplyConfig) ApplyKustomizeSDK(ctx context.Context, path string, namespace string) error {
	ac.Logger.Info("Applying kustomize resources from %s", path)

	// Build kustomized resources
	resources, err := ac.buildKustomize(path)
	if err != nil {
		return fmt.Errorf("failed to build kustomize: %w", err)
	}

	// Parse and apply resources
	decoder := yamlutil.NewYAMLOrJSONDecoder(bytes.NewReader(resources), 4096)

	for {
		var rawObj runtime.RawExtension
		if err := decoder.Decode(&rawObj); err != nil {
			if err == io.EOF {
				break
			}
			return fmt.Errorf("failed to decode resource: %w", err)
		}

		obj, gvk, err := yaml.NewDecodingSerializer(unstructured.UnstructuredJSONScheme).Decode(rawObj.Raw, nil, nil)
		if err != nil {
			ac.Logger.Warn("Failed to decode object: %v", err)
			continue
		}

		unstructuredMap, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
		if err != nil {
			ac.Logger.Warn("Failed to convert to unstructured: %v", err)
			continue
		}

		unstructuredObj := &unstructured.Unstructured{Object: unstructuredMap}

		// Override namespace if specified
		if namespace != "" && unstructuredObj.GetNamespace() == "" {
			unstructuredObj.SetNamespace(namespace)
		}

		// Apply the resource
		if err := ac.applyResource(ctx, unstructuredObj, gvk); err != nil {
			ac.Logger.Warn("Failed to apply resource %s/%s: %v",
				unstructuredObj.GetKind(), unstructuredObj.GetName(), err)
			continue
		}

		ac.Logger.Info("Applied %s/%s in namespace %s",
			unstructuredObj.GetKind(), unstructuredObj.GetName(), unstructuredObj.GetNamespace())
	}

	return nil
}

// buildKustomize builds kustomized resources
func (ac *ApplyConfig) buildKustomize(path string) ([]byte, error) {
	// Create kustomizer
	k := krusty.MakeKustomizer(krusty.MakeDefaultOptions())

	// Create filesystem
	fSys := filesys.MakeFsOnDisk()

	// Build resources
	resMap, err := k.Run(fSys, path)
	if err != nil {
		return nil, fmt.Errorf("kustomize build failed: %w", err)
	}

	// Convert to YAML
	yaml, err := resMap.AsYaml()
	if err != nil {
		return nil, fmt.Errorf("failed to convert to YAML: %w", err)
	}

	return yaml, nil
}

// applyResource applies a single resource
func (ac *ApplyConfig) applyResource(ctx context.Context, obj *unstructured.Unstructured, gvk *schema.GroupVersionKind) error {
	// Get resource mapping
	mapping, err := ac.Mapper.RESTMapping(gvk.GroupKind(), gvk.Version)
	if err != nil {
		return fmt.Errorf("failed to get REST mapping: %w", err)
	}

	// Get dynamic client for resource
	var dr dynamic.ResourceInterface
	if mapping.Scope.Name() == meta.RESTScopeNameNamespace {
		// Namespaced resource
		ns := obj.GetNamespace()
		if ns == "" {
			ns = "default"
		}
		dr = ac.Dynamic.Resource(mapping.Resource).Namespace(ns)
	} else {
		// Cluster-scoped resource
		dr = ac.Dynamic.Resource(mapping.Resource)
	}

	// Try to create first
	_, err = dr.Create(ctx, obj, metav1.CreateOptions{})
	if err != nil {
		if !k8serrors.IsAlreadyExists(err) {
			return fmt.Errorf("failed to create resource: %w", err)
		}

		// Resource exists, decide if we should update based on resource type
		kind := obj.GetKind()

		// Skip update for immutable resources
		if kind == "PersistentVolumeClaim" {
			// PVCs have immutable spec except for size
			return nil
		}

		// For PodDisruptionBudget, we need to get the current resourceVersion
		if kind == "PodDisruptionBudget" {
			current, err := dr.Get(ctx, obj.GetName(), metav1.GetOptions{})
			if err != nil {
				return fmt.Errorf("failed to get current PDB: %w", err)
			}
			obj.SetResourceVersion(current.GetResourceVersion())
		}

		// Try to update
		_, err = dr.Update(ctx, obj, metav1.UpdateOptions{})
		if err != nil {
			return fmt.Errorf("failed to update resource: %w", err)
		}
	}

	return nil
}

// DeleteKustomizeSDK deletes kustomize resources using the SDK
func (ac *ApplyConfig) DeleteKustomizeSDK(ctx context.Context, path string, namespace string) error {
	ac.Logger.Info("Deleting kustomize resources from %s", path)

	// Build kustomized resources
	resources, err := ac.buildKustomize(path)
	if err != nil {
		return fmt.Errorf("failed to build kustomize: %w", err)
	}

	// Parse and delete resources
	decoder := yamlutil.NewYAMLOrJSONDecoder(bytes.NewReader(resources), 4096)

	for {
		var rawObj runtime.RawExtension
		if err := decoder.Decode(&rawObj); err != nil {
			if err == io.EOF {
				break
			}
			return fmt.Errorf("failed to decode resource: %w", err)
		}

		obj, gvk, err := yaml.NewDecodingSerializer(unstructured.UnstructuredJSONScheme).Decode(rawObj.Raw, nil, nil)
		if err != nil {
			ac.Logger.Warn("Failed to decode object: %v", err)
			continue
		}

		unstructuredMap, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
		if err != nil {
			ac.Logger.Warn("Failed to convert to unstructured: %v", err)
			continue
		}

		unstructuredObj := &unstructured.Unstructured{Object: unstructuredMap}

		// Override namespace if specified
		if namespace != "" && unstructuredObj.GetNamespace() == "" {
			unstructuredObj.SetNamespace(namespace)
		}

		// Delete the resource
		if err := ac.deleteResource(ctx, unstructuredObj, gvk); err != nil {
			ac.Logger.Warn("Failed to delete resource %s/%s: %v",
				unstructuredObj.GetKind(), unstructuredObj.GetName(), err)
			continue
		}

		ac.Logger.Info("Deleted %s/%s in namespace %s",
			unstructuredObj.GetKind(), unstructuredObj.GetName(), unstructuredObj.GetNamespace())
	}

	return nil
}

// deleteResource deletes a single resource
func (ac *ApplyConfig) deleteResource(ctx context.Context, obj *unstructured.Unstructured, gvk *schema.GroupVersionKind) error {
	// Get resource mapping
	mapping, err := ac.Mapper.RESTMapping(gvk.GroupKind(), gvk.Version)
	if err != nil {
		return fmt.Errorf("failed to get REST mapping: %w", err)
	}

	// Get dynamic client for resource
	var dr dynamic.ResourceInterface
	if mapping.Scope.Name() == meta.RESTScopeNameNamespace {
		// Namespaced resource
		ns := obj.GetNamespace()
		if ns == "" {
			ns = "default"
		}
		dr = ac.Dynamic.Resource(mapping.Resource).Namespace(ns)
	} else {
		// Cluster-scoped resource
		dr = ac.Dynamic.Resource(mapping.Resource)
	}

	// Delete the resource
	err = dr.Delete(ctx, obj.GetName(), metav1.DeleteOptions{})
	if err != nil {
		return fmt.Errorf("failed to delete resource: %w", err)
	}

	return nil
}
