
export interface GroundTruthShape {
  type: string;
  center?: { x: number; y: number };
  bounding_box?: { x: number; y: number; width: number; height: number };
  area?: number;
  confidence_expected?: number;
  vertices?: { x: number; y: number }[];
  radius?: number;
  [key: string]: any;
}

export interface EvaluationMetrics {
  precision: number;
  recall: number;
  f1_score: number;
  average_iou: number;
  center_point_accuracy: number;
  area_accuracy: number;
  confidence_calibration: number;
  processing_time: number;
}


export function calculateIoU(box1: any, box2: any): number {
  const x1 = Math.max(box1.x, box2.x);
  const y1 = Math.max(box1.y, box2.y);
  const x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
  const y2 = Math.min(box1.y + box1.height, box2.y + box2.height);
  
  if (x2 <= x1 || y2 <= y1) return 0;
  
  const intersection = (x2 - x1) * (y2 - y1);
  const union = box1.width * box1.height + box2.width * box2.height - intersection;
  
  return intersection / union;
}


export function calculateDistance(p1: { x: number; y: number }, p2: { x: number; y: number }): number {
  return Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
}


export function evaluateDetection(detected: any[], groundTruth: GroundTruthShape[], imageName: string): EvaluationMetrics {
  const iouThreshold = 0.5;
  // Commented to fix deployment errors - these variables are reserved for future use
  // const centerThreshold = 10; 
  // const areaThreshold = 0.15;
  
  // Suppress unused parameter warning - imageName reserved for future logging/debugging
  void imageName; 
  
  let truePositives = 0;
  let totalIoU = 0;
  let totalCenterDistance = 0;
  let totalAreaError = 0;
  let confidenceErrors = 0;
  
  const matched = new Set();
  
  for (const detectedShape of detected) {
    let bestMatch = null;
    let bestIoU = 0;
    let bestIndex = -1;
    
    for (let i = 0; i < groundTruth.length; i++) {
      if (matched.has(i)) continue;
      
      const gtShape = groundTruth[i];
      if (detectedShape.type !== gtShape.type) continue;
      
      const iou = calculateIoU(detectedShape.boundingBox, gtShape.bounding_box);
      if (iou > bestIoU && iou > iouThreshold) {
        bestMatch = gtShape;
        bestIoU = iou;
        bestIndex = i;
      }
    }
    
    if (bestMatch) {
      matched.add(bestIndex);
      truePositives++;
      totalIoU += bestIoU;
      
      if (bestMatch.center && detectedShape.center) {
        const distance = calculateDistance(detectedShape.center, bestMatch.center);
        totalCenterDistance += distance;
      }
      
      if (bestMatch.area && detectedShape.area) {
        const areaError = Math.abs(detectedShape.area - bestMatch.area) / bestMatch.area;
        totalAreaError += areaError;
      }
      
      if (bestMatch.confidence_expected && detectedShape.confidence) {
        const confError = Math.abs(detectedShape.confidence - bestMatch.confidence_expected);
        confidenceErrors += confError;
      }
    }
  }
  
  // Handle edge case: if both detected and ground truth are empty, it's a perfect match
  const precision = detected.length > 0 ? truePositives / detected.length : (groundTruth.length === 0 ? 1 : 0);
  const recall = groundTruth.length > 0 ? truePositives / groundTruth.length : 1;
  const f1Score = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
  
  return {
    precision,
    recall,
    f1_score: f1Score,
    average_iou: truePositives > 0 ? totalIoU / truePositives : 0,
    center_point_accuracy: truePositives > 0 ? totalCenterDistance / truePositives : 0,
    area_accuracy: truePositives > 0 ? 1 - (totalAreaError / truePositives) : 0,
    confidence_calibration: truePositives > 0 ? 1 - (confidenceErrors / truePositives) : 0,
    processing_time: 0 
  };
}
