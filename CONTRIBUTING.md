# Contributing to QpiAI Quantum SDK

Thank you for your interest in the QpiAI Quantum SDK! We appreciate your support and welcome contributions from the community.

## About This Project

The QpiAI Quantum SDK is an open-source quantum computing framework distributed via PyPI. We welcome community contributions including bug reports, feature requests, code contributions, and documentation improvements.

## How You Can Contribute

### 1. Contributing Code

We welcome code contributions! To submit a code contribution:

1. **Fork the repository** on GitHub
2. **Create a feature branch** from `main`
3. **Make your changes** following the coding guidelines below
4. **Write tests** for your changes
5. **Submit a Merge Request** with a clear description of the changes

#### Coding Guidelines

- Follow PEP 8 style guidelines
- Use type hints where possible
- Write docstrings for all public methods and classes
- Ensure all existing tests pass before submitting
- Add tests for new functionality

#### Setting Up Development Environment

This project uses `uv` for dependency and environment management to guarantee consistent developer environments via `uv.lock`. If you do not have `uv` installed, you can install it via pip (`pip install uv`) or follow the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/).

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/quantum-sdk.git
cd quantum-sdk

# Create a virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode with dev dependencies from the lockfile
uv sync --extra dev

# Run tests
pytest tests/ -v
```

### 2. Reporting Bugs

Please refer to our [Reporting Bugs Guide](docs/REPORTING_BUGS.md) for detailed instructions on how to submit bug reports and reproducible examples.

### 3. Requesting Features

Please refer to our [Feature Requests Guide](docs/FEATURE_REQUESTS.md) to learn how to propose and discuss new features or enhancements.

### 4. Asking Questions

Have questions about using the SDK?

- **GitHub Discussions**: [https://github.com/qpiai/quantum-sdk/issues](https://github.com/qpiai/quantum-sdk/issues)
- **Email**: support@qcloud.qpiai.tech
- **Documentation**: [https://docs.qcloud.qpiai.tech/](https://docs.qcloud.qpiai.tech/)

Questions might include:
- How to implement specific quantum algorithms
- Best practices for circuit optimization
- Understanding SDK features and capabilities
- Integration with other frameworks

### 5. Sharing Your Work

We'd love to hear about projects built with QpiAI Quantum SDK!

- Share research papers, blog posts, or tutorials
- Let us know about educational materials you've created
- Tell us about applications you've built

Contact us at support@qcloud.qpiai.tech to share your work.

### 6. Documentation Feedback

Help us improve our documentation:

- Report unclear or missing documentation
- Suggest additional examples or tutorials
- Point out typos or errors
- Request explanations for advanced features

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment. When interacting with our team:

- Be respectful and considerate
- Use welcoming and inclusive language
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards others

### Response Times

We strive to respond to all inquiries within 48-72 hours during business days. Complex technical issues may require additional time to investigate.

### Priority of Issues

We prioritize issues in the following order:

1. Critical bugs affecting core functionality
2. Security vulnerabilities
3. Data loss or corruption issues
4. Non-critical bugs
5. Feature requests and enhancements

## Development Roadmap

We're continuously improving the QpiAI Quantum SDK. Areas of active development include:

- Enhanced quantum algorithm library (error correction, more variational algorithms)
- Additional backend and QPU support
- Performance optimizations for large-scale simulations
- Advanced visualization and analysis tools
- Noise modelling and error mitigation
- Expanded tutorials and documentation

## Getting Help

### Resources

- **Documentation**: [https://docs.qcloud.qpiai.tech/](https://docs.qcloud.qpiai.tech/)
- **PyPI Package**: [https://pypi.org/project/qpiai-quantum/](https://pypi.org/project/qpiai-quantum/)
- **Tutorial Notebooks**: Included in the `sdk_notebooks/` directory of the repository
- **Email Support**: support@qcloud.qpiai.tech

### Before Asking for Help

1. Check the documentation and API reference
2. Review the tutorial notebooks in `sdk_notebooks/`
3. Search existing GitHub issues
4. Try the latest version: `pip install --upgrade qpiai-quantum`

### Support Channels

- **GitHub Issues**: [https://github.com/qpiai/quantum-sdk/issues](https://github.com/qpiai/quantum-sdk/issues)
- **Technical & General Support**: support@qcloud.qpiai.tech

## Licensing

The QpiAI Quantum SDK is open-source software licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Recognition

We appreciate all community contributions! Significant contributors may be acknowledged in release notes or on our website (with permission).

- Bug reporters whose findings improve the SDK
- Code contributors who submit improvements and new features
- Users who provide detailed feature requests
- Educators creating learning materials
- Researchers publishing work using our SDK

## Stay Updated

Keep informed about SDK updates and new features:

- Watch the GitHub repository for new releases
- Watch the PyPI package for new releases
- Follow QpiAI for announcements

## Questions?

If you have any questions about contributing or using the QpiAI Quantum SDK, please don't hesitate to reach out to support@qcloud.qpiai.tech or open a GitHub issue.

---

**Thank you for being part of the QpiAI Quantum community!**

*Copyright © 2026 QpiAI. All rights reserved.*
