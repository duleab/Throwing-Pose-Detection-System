# Contributing to Violence Detection System

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, GPU/CPU)
- Screenshots or error messages

### Suggesting Features

Feature requests are welcome! Please include:
- Clear description of the feature
- Use case and benefits
- Possible implementation approach
- Any relevant examples

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/YourFeature
   ```
3. **Make your changes**
   - Follow the code style
   - Add tests if applicable
   - Update documentation
4. **Commit with clear messages**
   ```bash
   git commit -m "Add: Feature description"
   ```
5. **Push to your fork**
   ```bash
   git push origin feature/YourFeature
   ```
6. **Open a Pull Request**

## 📝 Code Style

### Python
- Follow PEP 8 guidelines
- Use type hints where possible
- Add docstrings to functions and classes
- Keep functions focused and small
- Use meaningful variable names

### Example:
```python
def detect_violence(frame: np.ndarray, threshold: float = 0.8) -> Dict[str, Any]:
    """
    Detect violence in a single frame.
    
    Args:
        frame: Input image as numpy array
        threshold: Confidence threshold for classification
        
    Returns:
        Dictionary containing detection results
    """
    # Implementation
    pass
```

### Documentation
- Use Markdown for documentation
- Include code examples
- Add screenshots where helpful
- Keep language clear and concise

## 🧪 Testing

Before submitting:
1. Test your changes locally
2. Ensure existing tests pass
3. Add new tests for new features
4. Test on both CPU and GPU if possible

## 📚 Documentation

Update documentation when:
- Adding new features
- Changing existing behavior
- Fixing bugs that affect usage
- Adding configuration options

## 🎯 Priority Areas

We especially welcome contributions in:
- Performance optimization
- New detection algorithms
- UI/UX improvements
- Documentation improvements
- Bug fixes
- Test coverage

## ❓ Questions

If you have questions:
- Check existing issues
- Read the documentation
- Create a new issue with the "question" label

## 🙏 Thank You!

Your contributions make this project better for everyone!
