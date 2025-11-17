# Sanskrit Ciphers
### Computational Puzzle-Solving for Ancient Buddhist Textual History

**Developer Team:**
- Omar El Aref
- Dylan Garner
- Muhammad Umar Khan
- Aswin Kuganesan
- Yousef Shahin

**Industry Advisor:** Dr. Shayne Clarke (McMaster Religious Studies)

**Project Start Date:** September 2025

**License:** GPL (General Public License)

**Repository:** [https://github.com/DylanG5/sanskrit-cipher](https://github.com/DylanG5/sanskrit-cipher)

---

## About This Project

This project addresses a critical challenge in Buddhist Studies: the textual history of Indian Buddhism is fragmented across thousands of manuscript folios, preserved only in partial, damaged, or scattered forms. Traditionally, scholars reconstruct these texts manually through paleographic study, transcription, and content comparison—a slow and error-prone process.

**Sanskrit Ciphers** is an ML-powered computational tool that automates the detection, matching, and transcription of ancient Buddhist manuscript fragments. By leveraging computer vision, machine learning, and OCR technology, we aim to enable large-scale reconstruction of Buddhist texts and accelerate scholarly research in digital humanities.

### Problem Statement

The lack of computational tools designed specifically for irregular, damaged, or arbitrarily oriented manuscript fragments significantly limits progress in reconstructing Buddhist textual history. Our system aims to:

- **Detect** edges, shapes, and damage patterns in manuscript fragments with high accuracy
- **Match** fragments based on probabilistic similarity measures (edge patterns, damage signatures, text content)
- **Transcribe** Sanskrit text using OCR models tuned for ancient scripts
- **Organize** fragment data in a searchable/sortable database with metadata and relationships
- **Provide** an intuitive interface for scholars to view, confirm, and annotate fragment matches

### Key Goals
- **Filtering From Dynamic Criteria:** Help Researchers filter through images with AI-powered criteria
- **Fragment Matching:** Demonstrate 70%+ precision in top-5 fragment match suggestions
- **Usability:** Enable non-technical scholars to use the system with <15 minutes of training
---

## Project Structure

The folders and files for this project are organized as follows:

```
sanskrit-cipher/
├── docs/                    # Comprehensive project documentation
│   ├── Design/             # Software architecture and detailed design documents
│   ├── DevelopmentPlan/    # Team workflow, milestones, and development strategy
│   ├── HazardAnalysis/     # Risk assessment and mitigation strategies
│   ├── ProblemStatementAndGoals/  # Project goals and success criteria
│   ├── Presentations/      # Demo and presentation materials
│   ├── SRS/                # Software Requirements Specification
│   ├── UserGuide/          # User documentation for scholars
│   ├── VnVPlan/            # Verification and Validation plan
│   └── VnVReport/          # Testing reports and results
│
├── refs/                    # Reference material, academic papers, and research
│
├── src/                     # Source code
│   └── web/                # Web application components
│       └── web-canvas/     # React/TypeScript frontend with interactive canvas
│
└── test/                    # Test cases and testing infrastructure
```
