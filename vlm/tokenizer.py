from typing import List, Dict


class DefectTokenizer:
    def encode(self, detections: List[Dict]) -> str:
        if not detections:
            return "Detected objects:\nNone"
        
        lines = ["Detected objects:"]
        for i, d in enumerate(detections, 1):
            x1, y1, x2, y2 = d.get("bbox", [0, 0, 0, 0])
            cx, cy = d.get("center", [0, 0])
            lines.append(
                f"{i}) class={d.get('defect_type', 'Unknown')}, "
                f"bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}), "
                f"center=({cx:.1f},{cy:.1f}), "
                f"conf={d.get('confidence', 0.0):.2f}, "
                f"severity={d.get('severity', 'Low')}"
            )
        return "\n".join(lines)
    
    def encode_compact(self, detections: List[Dict]) -> str:
        if not detections:
            return "No defects detected."
        by_type = {}
        for d in detections:
            t = d.get("defect_type", "Unknown")
            by_type.setdefault(t, []).append(d)
        lines = ["Detected defects:"]
        for t, ds in by_type.items():
            centers = [f"({d['center'][0]:.0f},{d['center'][1]:.0f})" for d in ds]
            avg_conf = sum(d.get("confidence", 0) for d in ds) / len(ds)
            lines.append(f"- {t}: {len(ds)} instance(s), centers={centers}, avg_conf={avg_conf:.2f}")
        return "\n".join(lines)
    
    def encode_for_counting(self, detections: List[Dict]) -> str:
        if not detections:
            return "Total defects: 0"
        counts = {}
        for d in detections:
            t = d.get("defect_type", "Unknown")
            counts[t] = counts.get(t, 0) + 1
        lines = [f"Total defects: {len(detections)}", "Breakdown by type:"]
        lines.extend(f"  - {t}: {c}" for t, c in sorted(counts.items()))
        return "\n".join(lines)
    
    def encode_for_location(self, detections: List[Dict]) -> str:
        if not detections:
            return "No defects detected."
        lines = ["Defect locations:"]
        for i, d in enumerate(detections, 1):
            cx, cy = d.get("center", [0, 0])
            x1, y1, x2, y2 = d.get("bbox", [0, 0, 0, 0])
            lines.append(f"{i}. {d.get('defect_type', 'Unknown')}: center=({cx:.1f}, {cy:.1f}), bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        return "\n".join(lines)
