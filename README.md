# MSc-Appendix-Materials
Code Availability and Author Contributions
The overall code structure is adapted from the framework proposed in
[7] Sun, X., Cheng, H., Li, J., Liu, B., and Guan, J. (2023). All in One: Multi-Task Prompting for Graph Neural Networks. Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 2120–2131.

Building upon this baseline, the author implemented task-specific adaptations and extensions to support the DGraphFin dataset and the requirements of dynamic financial graph modelling. In particular:
	•	dataset.py implements data loading and essential preprocessing tailored to the DGraphFin dataset.
	•	induction_timedecay.py implements a time-aware subgraph induction strategy used to reformulate the original node-level task into a graph-level learning setting.

These components were independently implemented and refined for this dissertation, while maintaining consistency with the original framework where appropriate.
