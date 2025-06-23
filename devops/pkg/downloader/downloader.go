// Package downloader provides file download functionality
package downloader

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/raja-aiml/sematic-cache/devops/internal/interfaces"
)

// FileDownloader implements the interfaces.FileDownloader interface
type FileDownloader struct {
	httpClient interfaces.HTTPClient
	logger     interfaces.Logger
}

// New creates a new file downloader
func New(httpClient interfaces.HTTPClient, logger interfaces.Logger) interfaces.FileDownloader {
	return &FileDownloader{
		httpClient: httpClient,
		logger:     logger,
	}
}

// Download downloads a file from URL to destination
func (d *FileDownloader) Download(ctx context.Context, url, destPath string) error {
	// Create destination directory if needed
	destDir := filepath.Dir(destPath)
	if err := os.MkdirAll(destDir, 0755); err != nil {
		return fmt.Errorf("failed to create destination directory: %w", err)
	}

	// Download file
	resp, err := d.httpClient.Get(ctx, url)
	if err != nil {
		return fmt.Errorf("failed to download file: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return fmt.Errorf("download failed with status: %d", resp.StatusCode)
	}

	// Create destination file
	out, err := os.Create(destPath)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer out.Close()

	// Copy data
	_, err = io.Copy(out, resp.Body)
	if err != nil {
		return fmt.Errorf("failed to save file: %w", err)
	}

	return nil
}

// DownloadWithProgress downloads a file with progress reporting
func (d *FileDownloader) DownloadWithProgress(ctx context.Context, url, destPath string, progress chan<- float64) error {
	defer close(progress)

	// Create destination directory if needed
	destDir := filepath.Dir(destPath)
	if err := os.MkdirAll(destDir, 0755); err != nil {
		return fmt.Errorf("failed to create destination directory: %w", err)
	}

	// Download file
	resp, err := d.httpClient.Get(ctx, url)
	if err != nil {
		return fmt.Errorf("failed to download file: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return fmt.Errorf("download failed with status: %d", resp.StatusCode)
	}

	// Get content length
	totalSize := resp.ContentLength
	if totalSize <= 0 {
		// No content length, fall back to regular download
		return d.Download(ctx, url, destPath)
	}

	// Create destination file
	out, err := os.Create(destPath)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer out.Close()

	// Create progress reader
	pr := &progressReader{
		reader:    resp.Body,
		totalSize: totalSize,
		progress:  progress,
	}

	// Copy data
	_, err = io.Copy(out, pr)
	if err != nil {
		return fmt.Errorf("failed to save file: %w", err)
	}

	return nil
}

// progressReader wraps an io.Reader and reports progress
type progressReader struct {
	reader      io.Reader
	totalSize   int64
	currentSize int64
	progress    chan<- float64
}

// Read implements io.Reader
func (pr *progressReader) Read(p []byte) (int, error) {
	n, err := pr.reader.Read(p)
	if n > 0 {
		pr.currentSize += int64(n)
		percentage := float64(pr.currentSize) / float64(pr.totalSize) * 100

		select {
		case pr.progress <- percentage:
		default:
			// Don't block if channel is full
		}
	}
	return n, err
}
