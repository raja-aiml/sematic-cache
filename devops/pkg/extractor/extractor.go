// Package extractor provides archive extraction functionality
package extractor

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/raja-aiml/sematic-cache/devops/internal/interfaces"
)

// ArchiveExtractor implements the interfaces.ArchiveExtractor interface
type ArchiveExtractor struct {
	logger interfaces.Logger
}

// New creates a new archive extractor
func New(logger interfaces.Logger) interfaces.ArchiveExtractor {
	return &ArchiveExtractor{
		logger: logger,
	}
}

// Extract extracts an archive to destination
func (e *ArchiveExtractor) Extract(src, dest string) error {
	// Determine archive type by extension
	switch {
	case strings.HasSuffix(src, ".tar.gz") || strings.HasSuffix(src, ".tgz"):
		return e.extractTarGz(src, dest)
	case strings.HasSuffix(src, ".tar"):
		return e.extractTar(src, dest)
	case strings.HasSuffix(src, ".zip"):
		return e.extractZip(src, dest)
	default:
		return fmt.Errorf("unsupported archive format: %s", src)
	}
}

// ExtractFile extracts a specific file from an archive
func (e *ArchiveExtractor) ExtractFile(src, dest, filename string) error {
	// Determine archive type by extension
	switch {
	case strings.HasSuffix(src, ".tar.gz") || strings.HasSuffix(src, ".tgz"):
		return e.extractFileFromTarGz(src, dest, filename)
	case strings.HasSuffix(src, ".tar"):
		return e.extractFileFromTar(src, dest, filename)
	case strings.HasSuffix(src, ".zip"):
		return e.extractFileFromZip(src, dest, filename)
	default:
		return fmt.Errorf("unsupported archive format: %s", src)
	}
}

// extractTarGz extracts a tar.gz archive
func (e *ArchiveExtractor) extractTarGz(src, dest string) error {
	file, err := os.Open(src)
	if err != nil {
		return fmt.Errorf("failed to open archive: %w", err)
	}
	defer file.Close()

	gzReader, err := gzip.NewReader(file)
	if err != nil {
		return fmt.Errorf("failed to create gzip reader: %w", err)
	}
	defer gzReader.Close()

	return e.extractTarReader(tar.NewReader(gzReader), dest)
}

// extractTar extracts a tar archive
func (e *ArchiveExtractor) extractTar(src, dest string) error {
	file, err := os.Open(src)
	if err != nil {
		return fmt.Errorf("failed to open archive: %w", err)
	}
	defer file.Close()

	return e.extractTarReader(tar.NewReader(file), dest)
}

// extractTarReader extracts from a tar reader
func (e *ArchiveExtractor) extractTarReader(tarReader *tar.Reader, dest string) error {
	for {
		header, err := tarReader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("failed to read tar header: %w", err)
		}

		// Construct the file path
		target := filepath.Join(dest, header.Name)

		// Ensure the target path is within dest (prevent path traversal)
		if !strings.HasPrefix(filepath.Clean(target), filepath.Clean(dest)) {
			return fmt.Errorf("invalid file path: %s", header.Name)
		}

		switch header.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, os.FileMode(header.Mode)); err != nil {
				return fmt.Errorf("failed to create directory: %w", err)
			}
		case tar.TypeReg:
			if err := e.extractTarFile(tarReader, target, os.FileMode(header.Mode)); err != nil {
				return fmt.Errorf("failed to extract file %s: %w", header.Name, err)
			}
		}
	}

	return nil
}

// extractTarFile extracts a single file from tar
func (e *ArchiveExtractor) extractTarFile(tarReader io.Reader, target string, mode os.FileMode) error {
	// Create directory if needed
	dir := filepath.Dir(target)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	// Create the file
	file, err := os.OpenFile(target, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()

	// Copy file contents
	if _, err := io.Copy(file, tarReader); err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}

	return nil
}

// extractZip extracts a zip archive
func (e *ArchiveExtractor) extractZip(src, dest string) error {
	reader, err := zip.OpenReader(src)
	if err != nil {
		return fmt.Errorf("failed to open zip file: %w", err)
	}
	defer reader.Close()

	for _, file := range reader.File {
		// Construct the file path
		target := filepath.Join(dest, file.Name)

		// Ensure the target path is within dest (prevent path traversal)
		if !strings.HasPrefix(filepath.Clean(target), filepath.Clean(dest)) {
			return fmt.Errorf("invalid file path: %s", file.Name)
		}

		if file.FileInfo().IsDir() {
			if err := os.MkdirAll(target, file.Mode()); err != nil {
				return fmt.Errorf("failed to create directory: %w", err)
			}
			continue
		}

		if err := e.extractZipFile(file, target); err != nil {
			return fmt.Errorf("failed to extract file %s: %w", file.Name, err)
		}
	}

	return nil
}

// extractZipFile extracts a single file from zip
func (e *ArchiveExtractor) extractZipFile(file *zip.File, target string) error {
	// Create directory if needed
	dir := filepath.Dir(target)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	// Open file in archive
	src, err := file.Open()
	if err != nil {
		return fmt.Errorf("failed to open file in archive: %w", err)
	}
	defer src.Close()

	// Create destination file
	dst, err := os.OpenFile(target, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, file.Mode())
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer dst.Close()

	// Copy file contents
	if _, err := io.Copy(dst, src); err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}

	return nil
}

// extractFileFromTarGz extracts a specific file from tar.gz
func (e *ArchiveExtractor) extractFileFromTarGz(src, dest, filename string) error {
	file, err := os.Open(src)
	if err != nil {
		return fmt.Errorf("failed to open archive: %w", err)
	}
	defer file.Close()

	gzReader, err := gzip.NewReader(file)
	if err != nil {
		return fmt.Errorf("failed to create gzip reader: %w", err)
	}
	defer gzReader.Close()

	return e.extractFileFromTarReader(tar.NewReader(gzReader), dest, filename)
}

// extractFileFromTar extracts a specific file from tar
func (e *ArchiveExtractor) extractFileFromTar(src, dest, filename string) error {
	file, err := os.Open(src)
	if err != nil {
		return fmt.Errorf("failed to open archive: %w", err)
	}
	defer file.Close()

	return e.extractFileFromTarReader(tar.NewReader(file), dest, filename)
}

// extractFileFromTarReader extracts a specific file from tar reader
func (e *ArchiveExtractor) extractFileFromTarReader(tarReader *tar.Reader, dest, filename string) error {
	for {
		header, err := tarReader.Next()
		if err == io.EOF {
			return fmt.Errorf("file not found in archive: %s", filename)
		}
		if err != nil {
			return fmt.Errorf("failed to read tar header: %w", err)
		}

		if header.Name == filename || filepath.Base(header.Name) == filename {
			target := filepath.Join(dest, filepath.Base(filename))
			return e.extractTarFile(tarReader, target, os.FileMode(header.Mode))
		}
	}
}

// extractFileFromZip extracts a specific file from zip
func (e *ArchiveExtractor) extractFileFromZip(src, dest, filename string) error {
	reader, err := zip.OpenReader(src)
	if err != nil {
		return fmt.Errorf("failed to open zip file: %w", err)
	}
	defer reader.Close()

	for _, file := range reader.File {
		if file.Name == filename || filepath.Base(file.Name) == filename {
			target := filepath.Join(dest, filepath.Base(filename))
			return e.extractZipFile(file, target)
		}
	}

	return fmt.Errorf("file not found in archive: %s", filename)
}
