package build

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"testing"
	"time"

	"github.com/raja-aiml/sematic-cache/deploy/local/pkg/interfaces"
)

// Mock implementations
type mockBuildProvider struct {
	buildErr       error
	pushErr        error
	tagErr         error
	pullErr        error
	existsResult   bool
	existsErr      error
	removeErr      error
	listResult     []ImageInfo
	listErr        error
	buildCallCount int
}

func (m *mockBuildProvider) Build(ctx context.Context, options ProviderBuildOptions) error {
	m.buildCallCount++
	return m.buildErr
}

func (m *mockBuildProvider) Push(ctx context.Context, image string, auth *AuthConfig) error {
	return m.pushErr
}

func (m *mockBuildProvider) Tag(ctx context.Context, source, target string) error {
	return m.tagErr
}

func (m *mockBuildProvider) Pull(ctx context.Context, image string, auth *AuthConfig) error {
	return m.pullErr
}

func (m *mockBuildProvider) ImageExists(ctx context.Context, image string) (bool, error) {
	return m.existsResult, m.existsErr
}

func (m *mockBuildProvider) RemoveImage(ctx context.Context, image string) error {
	return m.removeErr
}

func (m *mockBuildProvider) ListImages(ctx context.Context) ([]ImageInfo, error) {
	return m.listResult, m.listErr
}

type mockBuildCache struct {
	getResult    *CacheEntry
	getErr       error
	setErr       error
	deleteErr    error
	clearErr     error
	setCalled    bool
	deleteCalled bool
}

func (m *mockBuildCache) Get(ctx context.Context, key string) (*CacheEntry, error) {
	return m.getResult, m.getErr
}

func (m *mockBuildCache) Set(ctx context.Context, key string, entry *CacheEntry) error {
	m.setCalled = true
	return m.setErr
}

func (m *mockBuildCache) Delete(ctx context.Context, key string) error {
	m.deleteCalled = true
	return m.deleteErr
}

func (m *mockBuildCache) Clear(ctx context.Context) error {
	return m.clearErr
}

func TestNewBuilder(t *testing.T) {
	provider := &mockBuildProvider{}
	cache := &mockBuildCache{}
	logger := slog.Default()
	config := BuilderConfig{
		EnableCache:     true,
		CacheTTL:        1 * time.Hour,
		DefaultPlatform: "linux/amd64",
	}

	builder := NewBuilder(provider, cache, logger, config)

	if builder == nil {
		t.Fatal("NewBuilder returned nil")
	}

	if builder.provider != provider {
		t.Error("Builder provider mismatch")
	}

	if builder.cache != cache {
		t.Error("Builder cache mismatch")
	}

	if builder.config.EnableCache != config.EnableCache {
		t.Error("Builder config mismatch")
	}
}

func TestBuilder_Build(t *testing.T) {
	tests := []struct {
		name          string
		options       interfaces.BuildOptions
		providerErr   error
		cacheEnabled  bool
		cachedEntry   *CacheEntry
		wantErr       bool
		wantBuildCall bool
	}{
		{
			name: "successful_build",
			options: interfaces.BuildOptions{
				ImageName:      "test:latest",
				DockerfilePath: "Dockerfile",
				BuildContext:   ".",
			},
			providerErr:   nil,
			cacheEnabled:  false,
			wantErr:       false,
			wantBuildCall: true,
		},
		{
			name: "build_with_error",
			options: interfaces.BuildOptions{
				ImageName:      "test:latest",
				DockerfilePath: "Dockerfile",
				BuildContext:   ".",
			},
			providerErr:   errors.New("build failed"),
			cacheEnabled:  false,
			wantErr:       true,
			wantBuildCall: true,
		},
		{
			name: "cached_build",
			options: interfaces.BuildOptions{
				ImageName:      "test:latest",
				DockerfilePath: "Dockerfile",
				BuildContext:   ".",
			},
			cacheEnabled: true,
			cachedEntry: &CacheEntry{
				ImageID:   "cached-id",
				Tags:      []string{"test:latest"},
				ExpiresAt: time.Now().Add(1 * time.Hour),
			},
			wantErr:       false,
			wantBuildCall: false,
		},
		{
			name: "expired_cache",
			options: interfaces.BuildOptions{
				ImageName:      "test:latest",
				DockerfilePath: "Dockerfile",
				BuildContext:   ".",
			},
			cacheEnabled: true,
			cachedEntry: &CacheEntry{
				ImageID:   "cached-id",
				Tags:      []string{"test:old"},
				ExpiresAt: time.Now().Add(-1 * time.Hour),
			},
			wantErr:       false,
			wantBuildCall: true,
		},
		{
			name: "no_cache_flag",
			options: interfaces.BuildOptions{
				ImageName:      "test:latest",
				DockerfilePath: "Dockerfile",
				BuildContext:   ".",
				NoCache:        true,
			},
			cacheEnabled: true,
			cachedEntry: &CacheEntry{
				ImageID:   "cached-id",
				Tags:      []string{"test:latest"},
				ExpiresAt: time.Now().Add(1 * time.Hour),
			},
			wantErr:       false,
			wantBuildCall: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := &mockBuildProvider{
				buildErr: tt.providerErr,
				listResult: []ImageInfo{
					{
						ID:   "test-id",
						Tags: []string{tt.options.ImageName},
						Size: 1024,
					},
				},
			}
			cache := &mockBuildCache{
				getResult: tt.cachedEntry,
			}
			logger := slog.Default()
			config := BuilderConfig{
				EnableCache: tt.cacheEnabled,
				CacheTTL:    1 * time.Hour,
			}

			builder := NewBuilder(provider, cache, logger, config)
			ctx := context.Background()

			err := builder.Build(ctx, tt.options)

			if (err != nil) != tt.wantErr {
				t.Errorf("Build() error = %v, wantErr %v", err, tt.wantErr)
			}

			if provider.buildCallCount > 0 != tt.wantBuildCall {
				t.Errorf("Build() buildCallCount = %d, wantBuildCall %v", provider.buildCallCount, tt.wantBuildCall)
			}

			if tt.cacheEnabled && !tt.wantErr && tt.wantBuildCall && !cache.setCalled {
				t.Error("Build() should have cached the image")
			}
		})
	}
}

func TestBuilder_Push(t *testing.T) {
	tests := []struct {
		name    string
		image   string
		pushErr error
		wantErr bool
	}{
		{
			name:    "successful_push",
			image:   "test:latest",
			pushErr: nil,
			wantErr: false,
		},
		{
			name:    "push_error",
			image:   "test:latest",
			pushErr: errors.New("push failed"),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := &mockBuildProvider{
				pushErr: tt.pushErr,
			}
			cache := &mockBuildCache{}
			logger := slog.Default()
			config := BuilderConfig{}

			builder := NewBuilder(provider, cache, logger, config)
			ctx := context.Background()

			err := builder.Push(ctx, tt.image)

			if (err != nil) != tt.wantErr {
				t.Errorf("Push() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestBuilder_Tag(t *testing.T) {
	tests := []struct {
		name    string
		source  string
		target  string
		tagErr  error
		wantErr bool
	}{
		{
			name:    "successful_tag",
			source:  "test:latest",
			target:  "test:v1.0",
			tagErr:  nil,
			wantErr: false,
		},
		{
			name:    "tag_error",
			source:  "test:latest",
			target:  "test:v1.0",
			tagErr:  errors.New("tag failed"),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := &mockBuildProvider{
				tagErr: tt.tagErr,
			}
			cache := &mockBuildCache{}
			logger := slog.Default()
			config := BuilderConfig{}

			builder := NewBuilder(provider, cache, logger, config)
			ctx := context.Background()

			err := builder.Tag(ctx, tt.source, tt.target)

			if (err != nil) != tt.wantErr {
				t.Errorf("Tag() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestBuilder_Pull(t *testing.T) {
	tests := []struct {
		name         string
		image        string
		exists       bool
		existsErr    error
		pullErr      error
		wantErr      bool
		wantPullCall bool
	}{
		{
			name:         "successful_pull",
			image:        "test:latest",
			exists:       false,
			pullErr:      nil,
			wantErr:      false,
			wantPullCall: true,
		},
		{
			name:         "image_exists",
			image:        "test:latest",
			exists:       true,
			wantErr:      false,
			wantPullCall: false,
		},
		{
			name:         "pull_error",
			image:        "test:latest",
			exists:       false,
			pullErr:      errors.New("pull failed"),
			wantErr:      true,
			wantPullCall: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := &mockBuildProvider{
				existsResult: tt.exists,
				existsErr:    tt.existsErr,
				pullErr:      tt.pullErr,
			}
			cache := &mockBuildCache{}
			logger := slog.Default()
			config := BuilderConfig{}

			builder := NewBuilder(provider, cache, logger, config)
			ctx := context.Background()

			err := builder.Pull(ctx, tt.image)

			if (err != nil) != tt.wantErr {
				t.Errorf("Pull() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestBuilder_Exists(t *testing.T) {
	tests := []struct {
		name      string
		image     string
		exists    bool
		existsErr error
		wantErr   bool
	}{
		{
			name:    "image_exists",
			image:   "test:latest",
			exists:  true,
			wantErr: false,
		},
		{
			name:    "image_not_exists",
			image:   "test:latest",
			exists:  false,
			wantErr: false,
		},
		{
			name:      "exists_error",
			image:     "test:latest",
			existsErr: errors.New("check failed"),
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := &mockBuildProvider{
				existsResult: tt.exists,
				existsErr:    tt.existsErr,
			}
			cache := &mockBuildCache{}
			logger := slog.Default()
			config := BuilderConfig{}

			builder := NewBuilder(provider, cache, logger, config)
			ctx := context.Background()

			exists, err := builder.Exists(ctx, tt.image)

			if (err != nil) != tt.wantErr {
				t.Errorf("Exists() error = %v, wantErr %v", err, tt.wantErr)
			}

			if !tt.wantErr && exists != tt.exists {
				t.Errorf("Exists() = %v, want %v", exists, tt.exists)
			}
		})
	}
}

func TestBuilder_Remove(t *testing.T) {
	tests := []struct {
		name         string
		image        string
		removeErr    error
		cacheEnabled bool
		wantErr      bool
	}{
		{
			name:         "successful_remove",
			image:        "test:latest",
			removeErr:    nil,
			cacheEnabled: false,
			wantErr:      false,
		},
		{
			name:         "remove_with_cache",
			image:        "test:latest",
			removeErr:    nil,
			cacheEnabled: true,
			wantErr:      false,
		},
		{
			name:      "remove_error",
			image:     "test:latest",
			removeErr: errors.New("remove failed"),
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := &mockBuildProvider{
				removeErr: tt.removeErr,
			}
			cache := &mockBuildCache{}
			logger := slog.Default()
			config := BuilderConfig{
				EnableCache: tt.cacheEnabled,
			}

			builder := NewBuilder(provider, cache, logger, config)
			ctx := context.Background()

			err := builder.Remove(ctx, tt.image)

			if (err != nil) != tt.wantErr {
				t.Errorf("Remove() error = %v, wantErr %v", err, tt.wantErr)
			}

			if tt.cacheEnabled && !tt.wantErr && !cache.deleteCalled {
				t.Error("Remove() should have deleted from cache")
			}
		})
	}
}

func TestBuilder_List(t *testing.T) {
	tests := []struct {
		name       string
		pattern    string
		listResult []ImageInfo
		listErr    error
		want       []string
		wantErr    bool
	}{
		{
			name:    "list_all",
			pattern: "",
			listResult: []ImageInfo{
				{ID: "1", Tags: []string{"app:latest", "app:v1"}},
				{ID: "2", Tags: []string{"db:latest"}},
			},
			want:    []string{"app:latest", "app:v1", "db:latest"},
			wantErr: false,
		},
		{
			name:    "list_with_pattern",
			pattern: "app:*",
			listResult: []ImageInfo{
				{ID: "1", Tags: []string{"app:latest", "app:v1"}},
				{ID: "2", Tags: []string{"db:latest"}},
			},
			want:    []string{"app:latest", "app:v1"},
			wantErr: false,
		},
		{
			name:    "list_error",
			pattern: "",
			listErr: errors.New("list failed"),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := &mockBuildProvider{
				listResult: tt.listResult,
				listErr:    tt.listErr,
			}
			cache := &mockBuildCache{}
			logger := slog.Default()
			config := BuilderConfig{}

			builder := NewBuilder(provider, cache, logger, config)
			ctx := context.Background()

			got, err := builder.List(ctx, tt.pattern)

			if (err != nil) != tt.wantErr {
				t.Errorf("List() error = %v, wantErr %v", err, tt.wantErr)
			}

			if !tt.wantErr && len(got) != len(tt.want) {
				t.Errorf("List() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMatchesPattern(t *testing.T) {
	tests := []struct {
		name    string
		s       string
		pattern string
		want    bool
	}{
		{
			name:    "exact_match",
			s:       "test:latest",
			pattern: "test:latest",
			want:    true,
		},
		{
			name:    "wildcard_all",
			s:       "test:latest",
			pattern: "*",
			want:    true,
		},
		{
			name:    "prefix_match",
			s:       "test:latest",
			pattern: "test:*",
			want:    true,
		},
		{
			name:    "suffix_match",
			s:       "test:latest",
			pattern: "*:latest",
			want:    true,
		},
		{
			name:    "no_match",
			s:       "test:latest",
			pattern: "prod:*",
			want:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := matchesPattern(tt.s, tt.pattern); got != tt.want {
				t.Errorf("matchesPattern() = %v, want %v", got, tt.want)
			}
		})
	}
}

// Benchmark tests
func BenchmarkBuilder_Build(b *testing.B) {
	provider := &mockBuildProvider{}
	cache := &mockBuildCache{}
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	config := BuilderConfig{
		EnableCache: true,
		CacheTTL:    1 * time.Hour,
	}

	builder := NewBuilder(provider, cache, logger, config)
	ctx := context.Background()
	options := interfaces.BuildOptions{
		ImageName:      "test:latest",
		DockerfilePath: "Dockerfile",
		BuildContext:   ".",
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = builder.Build(ctx, options)
	}
}

func BenchmarkGenerateCacheKey(b *testing.B) {
	provider := &mockBuildProvider{}
	cache := &mockBuildCache{}
	logger := slog.Default()
	config := BuilderConfig{}

	builder := NewBuilder(provider, cache, logger, config)
	options := interfaces.BuildOptions{
		ImageName:      "test:latest",
		DockerfilePath: "Dockerfile",
		BuildContext:   ".",
		BuildArgs: map[string]string{
			"VERSION": "1.0",
			"ENV":     "prod",
		},
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = builder.generateCacheKey(options)
	}
}
