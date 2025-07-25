# 🌬️ Zephyra - Intelligent Object Detection & Counting System

**Zephyra** is a lightweight AI-powered system that analyzes images, detects objects using deep learning, and provides a clear summary of object counts. Designed with simplicity and usability in mind, Zephyra leverages powerful vision models and a sleek **Gradio interface** to deliver fast and accurate scene analysis.

---

## 🧠 What Zephyra Does

✅ Detects multiple objects in an image  
✅ Counts how many instances of each object type are present  
✅ Presents results with bounding box overlays  
✅ Provides a **clean Gradio web UI** for easy image upload and results viewing

> ⚠️ Note: **Zephyra does not generate scene descriptions**. It focuses purely on **object detection and counting**.

---

## 🛠️ Technologies Used

| Purpose               | Tool/Framework           |
|------------------------|--------------------------|
| Object Detection       | YOLOv5 (PyTorch)          |
| Image Processing       | OpenCV                    |
| Web Interface          | Gradio                    |
| Data Structures        | Python (dicts, lists)     |
| Visualization          | Matplotlib, PIL           |



---



## 📊 Sample Output

**Input Image**:  
Input an image from your system

**Detected and Counted Output**:
```
- Person: 4
- Car: 2
- Traffic Light: 1
```

Bounding boxes are drawn around detected objects in the result image.

---

## 🎯 Future Plans

- Add object confidence thresholds for user control
- Describes the complete relations in the image
- Export results to text or JSON  
- Batch image support  
- Option to auto-save annotated images

---


## 👤 Author

**Vivek K K**  
🔗 [LinkedIn](https://www.linkedin.com/in/vivek-k-k)  
📧 [Email](mailto:vivek.kk2024aiml@sece.ac.in)

---

> _"Zephyra – See more, count smarter."_ 🧠📷
