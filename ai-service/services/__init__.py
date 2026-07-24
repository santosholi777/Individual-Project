"""AI and business services for DeepVisionAttend.

Each module owns one responsibility of the pipeline:

* :mod:`services.engine` — downloads and resolves the pre-trained model pack.
* :mod:`services.detector` — locates faces (SCRFD).
* :mod:`services.embedder` — aligns and embeds faces (ArcFace).
* :mod:`services.matcher` — resolves identity by cosine similarity.
* :mod:`services.registration_service` — enrols students.
* :mod:`services.recognition_service` — orchestrates the full pipeline.
* :mod:`services.attendance_service` — applies attendance policy.
"""
