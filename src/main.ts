import "./style.css";
import { SelectionManager } from "./ui-utils.js";
import { EvaluationManager } from "./evaluation-manager.js";

export interface Point {
  x: number;
  y: number;
}

export interface DetectedShape {
  type: "circle" | "triangle" | "rectangle" | "square" | "pentagon" | "star";
  confidence: number;
  boundingBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  center: Point;
  area: number;
}

export interface DetectionResult {
  shapes: DetectedShape[];
  processingTime: number;
  imageWidth: number;
  imageHeight: number;
}

export class ShapeDetector {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
  }

  async detectShapes(imageData: ImageData): Promise<DetectionResult> {
    const startTime = performance.now();
    const { width, height, data } = imageData;

    try {
      const grayscale = this.toGrayscale(data, width, height);
      const otsuThreshold = this.otsuThreshold(grayscale);
      let otsuForeground = 0;
      for (let i = 0; i < grayscale.length; i++) {
        if (grayscale[i] < otsuThreshold) otsuForeground++;
      }
      const otsuRatio = otsuForeground / (width * height);
      const threshold = (otsuRatio > 0.05 && otsuRatio < 0.50) ? otsuThreshold : 128;
      
      const binary = new Uint8ClampedArray(width * height);
      for (let i = 0; i < grayscale.length; i++) {
        binary[i] = grayscale[i] < threshold ? 255 : 0;
      }

      const contours = this.traceContours(binary, width, height);
      const shapes: DetectedShape[] = [];
      const minArea = Math.max(50, (width * height) * 0.0005);
      const maxArea = (width * height) * 0.9;

      for (const contour of contours) {
        if (contour.length < 10) continue;

        const area = this.calculateArea(contour);
        if (area < minArea || area > maxArea) continue;

        const perimeter = this.calculatePerimeter(contour);
        const bbox = this.getBoundingBox(contour);
        const aspectRatio = bbox.width > 0 ? bbox.height / bbox.width : 1;
        const normalizedAspect = Math.min(aspectRatio, 1 / aspectRatio);
        
        if (normalizedAspect < 0.15) continue;
        
        let bestShape: { type: DetectedShape["type"]; confidence: number; approx: Point[] } | null = null;
        let bestScore = 0;
        
        for (const eps of [0.008, 0.01, 0.012, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]) {
          const approx = this.approximatePolygon(contour, eps * perimeter);
          if (approx.length < 3) continue;

          const shapeInfo = this.classifyShapeAdvanced(approx, area, perimeter, this.getBoundingBox(approx));
          if (!shapeInfo) continue;

          const v = approx.length;
          const score = shapeInfo.confidence + (v === 3 ? 0.30 : v === 4 ? 0.25 : v === 5 ? 0.10 : v >= 8 && v <= 12 ? 0.08 : 0);
          
          if (score > bestScore) {
            bestShape = { ...shapeInfo, approx };
            bestScore = score;
          }
        }

        if (bestShape && bestShape.confidence >= 0.55) {
          const bbox = this.getBoundingBox(bestShape.approx);
          const center = this.calculateCentroid(bestShape.approx);
          shapes.push({
            type: bestShape.type,
            confidence: bestShape.confidence,
            boundingBox: bbox,
            center: center,
            area: Math.round(area),
          });
        }
      }

      const finalShapes = this.nonMaxSuppression(shapes, 0.5);
      const processingTime = performance.now() - startTime;

      return {
        shapes: finalShapes,
        processingTime,
        imageWidth: width,
        imageHeight: height,
      };
    } catch (error) {
      return {
        shapes: [],
        processingTime: performance.now() - startTime,
        imageWidth: width,
        imageHeight: height,
      };
    }
  }

  private toGrayscale(data: Uint8ClampedArray, width: number, height: number): Float32Array {
    const gray = new Float32Array(width * height);
    for (let i = 0; i < width * height; i++) {
      const idx = i * 4;
      gray[i] = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
    }
    return gray;
  }

  private traceContours(binary: Uint8ClampedArray, width: number, height: number): Point[][] {
    const visited = new Uint8ClampedArray(width * height);
    const contours: Point[][] = [];

    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = y * width + x;
        
        if (binary[idx] === 255 && !visited[idx]) {
          let isEdge = false;
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              if (dx === 0 && dy === 0) continue;
              const nx = x + dx;
              const ny = y + dy;
              if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                const nidx = ny * width + nx;
                if (binary[nidx] === 0) {
                  isEdge = true;
                  break;
                }
              }
            }
            if (isEdge) break;
          }
          
          if (isEdge) {
            const contour = this.borderFollowing(binary, visited, x, y, width, height);
            
            if (contour.length >= 10) {
              contours.push(contour);
            }
          }
        }
      }
    }

    return contours;
  }

  private borderFollowing(
    binary: Uint8ClampedArray,
    visited: Uint8ClampedArray,
    startX: number,
    startY: number,
    width: number,
    height: number
  ): Point[] {
    const contour: Point[] = [];
    const directions = [
      { x: 1, y: 0 },   // right
      { x: 1, y: -1 },  // top-right
      { x: 0, y: -1 },  // top
      { x: -1, y: -1 }, // top-left
      { x: -1, y: 0 },  // left
      { x: -1, y: 1 },  // bottom-left
      { x: 0, y: 1 },   // bottom
      { x: 1, y: 1 }    // bottom-right
    ];

    let current = { x: startX, y: startY };
    const start = { x: startX, y: startY };
    let dir = 0;
    let iterations = 0;
    const maxIterations = width * height;

    do {
      contour.push({ ...current });
      visited[current.y * width + current.x] = 1;

      let found = false;
      for (let i = 0; i < 8; i++) {
        const checkDir = (dir + i) % 8;
        const nx = current.x + directions[checkDir].x;
        const ny = current.y + directions[checkDir].y;

        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
          const nidx = ny * width + nx;
          if (binary[nidx] === 255) {
            current = { x: nx, y: ny };
            dir = (checkDir + 6) % 8;
            found = true;
            break;
          }
        }
      }

      if (!found) break;
      iterations++;
    } while (!(current.x === start.x && current.y === start.y) && iterations < maxIterations);

    return contour;
  }

  private calculatePerimeter(points: Point[]): number {
    let perimeter = 0;
    const n = points.length;
    
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      const dx = points[j].x - points[i].x;
      const dy = points[j].y - points[i].y;
      perimeter += Math.sqrt(dx * dx + dy * dy);
    }
    
    return perimeter;
  }

  private classifyShapeAdvanced(
    approx: Point[],
    area: number,
    perimeter: number,
    bbox: { x: number; y: number; width: number; height: number }
  ): { type: DetectedShape["type"]; confidence: number } | null {
    const vertices = approx.length;
    const circularity = (4 * Math.PI * area) / (perimeter * perimeter);
    const aspectRatio = bbox.width > 0 ? bbox.height / bbox.width : 1;
    const normalizedAspect = Math.min(aspectRatio, 1 / aspectRatio);
    const hull = this.convexHull(approx);
    const hullArea = this.calculateArea(hull);
    const convexity = hullArea > 0 ? area / hullArea : 0;
    const bboxArea = bbox.width * bbox.height;
    const extent = bboxArea > 0 ? area / bboxArea : 0;

    if (convexity > 0.75) {
      if (vertices === 3) {
        return { type: "triangle", confidence: Math.min(0.95, 0.80 + convexity * 0.15) };
      }
      if (vertices === 4) {
        if (circularity < 0.65 && extent < 0.60) {
          return { type: "triangle", confidence: Math.min(0.85, 0.70 + convexity * 0.15) };
        }
        return { type: "rectangle", confidence: Math.min(0.95, 0.80 + convexity * 0.15) };
      }
      if (vertices === 5) {
        const isSquareLike = circularity > 0.75 && extent > 0.95 && normalizedAspect > 0.95;
        const isRectLike = extent > 0.85 && (circularity > 0.50 && circularity < 0.80);
        const isRotatedRect = circularity > 0.50 && circularity < 0.70 && convexity > 0.90 && extent < 0.70;
        
        if (isSquareLike || isRectLike || isRotatedRect) {
          return { type: "rectangle", confidence: Math.min(0.90, 0.75 + convexity * 0.15) };
        }
        return { type: "pentagon", confidence: Math.min(0.95, 0.75 + convexity * 0.20) };
      }
      if (vertices === 6) {
        return { type: "pentagon", confidence: Math.min(0.88, 0.70 + convexity * 0.18) };
      }
    }

    if (circularity > 0.75 && extent > 0.65) {
      return { type: "circle", confidence: Math.min(0.95, 0.70 + circularity * 0.25) };
    }

    if (vertices >= 8 && vertices <= 12 && convexity >= 0.30 && convexity <= 0.75) {
      const center = this.calculateCentroid(approx);
      const distances: number[] = [];
      for (const point of approx) {
        const dx = point.x - center.x;
        const dy = point.y - center.y;
        distances.push(Math.sqrt(dx * dx + dy * dy));
      }

      let alternations = 0;
      for (let i = 0; i < distances.length; i++) {
        const prev = distances[(i - 1 + distances.length) % distances.length];
        const curr = distances[i];
        const next = distances[(i + 1) % distances.length];
        
        if ((curr > prev && curr > next) || (curr < prev && curr < next)) {
          alternations++;
        }
      }
      
      const alternationScore = alternations / distances.length;
      const sorted = distances.slice().sort((a, b) => a - b);
      const shortDist = sorted[Math.floor(sorted.length / 4)];
      const longDist = sorted[Math.floor(3 * sorted.length / 4)];
      const alternationRatio = longDist / (shortDist || 1);

      if (alternationRatio > 1.15 && alternationScore > 0.5) {
        return { type: "star", confidence: Math.min(0.90, 0.60 + (alternationRatio - 1) * 0.20 + alternationScore * 0.10) };
      }
    }

    if (circularity > 0.60 && extent > 0.55) {
      return { type: "circle", confidence: 0.60 };
    }
    if (vertices === 3 && convexity > 0.60) {
      return { type: "triangle", confidence: Math.min(0.75, 0.55 + convexity * 0.20) };
    }
    if (vertices === 4 && convexity > 0.65) {
      return { type: "rectangle", confidence: 0.60 };
    }
    if (vertices === 5 && convexity > 0.65) {
      return { type: "pentagon", confidence: 0.60 };
    }

    return null;
  }

  private otsuThreshold(gray: Float32Array): number {
    const histogram = new Array(256).fill(0);
    let maxVal = 0;
    
    for (let i = 0; i < gray.length; i++) {
      const val = Math.min(255, Math.max(0, Math.round(gray[i])));
      histogram[val]++;
      maxVal = Math.max(maxVal, val);
    }

    const total = gray.length;
    for (let i = 0; i <= maxVal; i++) {
      histogram[i] /= total;
    }

    let bestThreshold = 0;
    let maxVariance = 0;

    for (let t = 0; t <= maxVal; t++) {
      let w0 = 0, w1 = 0;
      let mu0 = 0, mu1 = 0;

      for (let i = 0; i <= t; i++) {
        w0 += histogram[i];
        mu0 += i * histogram[i];
      }
      if (w0 > 0) mu0 /= w0;

      for (let i = t + 1; i <= maxVal; i++) {
        w1 += histogram[i];
        mu1 += i * histogram[i];
      }
      if (w1 > 0) mu1 /= w1;

      const variance = w0 * w1 * (mu0 - mu1) * (mu0 - mu1);
      
      if (variance > maxVariance) {
        maxVariance = variance;
        bestThreshold = t;
      }
    }

    return bestThreshold;
  }

  loadImage(file: File): Promise<ImageData> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        this.canvas.width = img.width;
        this.canvas.height = img.height;
        this.ctx.drawImage(img, 0, 0);
        const imageData = this.ctx.getImageData(0, 0, img.width, img.height);
        resolve(imageData);
      };
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  }

  // ============ GEOMETRIC UTILITIES ============

  private calculateArea(contour: Point[]): number {
    let area = 0;
    for (let i = 0; i < contour.length; i++) {
      const j = (i + 1) % contour.length;
      area += contour[i].x * contour[j].y;
      area -= contour[j].x * contour[i].y;
    }
    return Math.abs(area) / 2;
  }

  private calculateCentroid(contour: Point[]): Point {
    let sumX = 0, sumY = 0;
    for (const point of contour) {
      sumX += point.x;
      sumY += point.y;
    }
    return {
      x: Math.round(sumX / contour.length),
      y: Math.round(sumY / contour.length),
    };
  }

  private getBoundingBox(contour: Point[]): { x: number; y: number; width: number; height: number } {
    let minX = Infinity, minY = Infinity;
    let maxX = -Infinity, maxY = -Infinity;

    for (const point of contour) {
      minX = Math.min(minX, point.x);
      minY = Math.min(minY, point.y);
      maxX = Math.max(maxX, point.x);
      maxY = Math.max(maxY, point.y);
    }

    return {
      x: minX,
      y: minY,
      width: maxX - minX,
      height: maxY - minY,
    };
  }

  private approximatePolygon(contour: Point[], epsilon: number): Point[] {
    if (contour.length < 3) return contour;

    const rdp = (points: Point[], start: number, end: number, eps: number): Point[] => {
      if (end - start < 2) return [points[start]];

      let maxDist = 0;
      let maxIndex = start;
      const lineStart = points[start];
      const lineEnd = points[end];

      for (let i = start + 1; i < end; i++) {
        const dist = this.perpendicularDistance(points[i], lineStart, lineEnd);
        if (dist > maxDist) {
          maxDist = dist;
          maxIndex = i;
        }
      }

      if (maxDist > eps) {
        const left = rdp(points, start, maxIndex, eps);
        const right = rdp(points, maxIndex, end, eps);
        return [...left, ...right];
      } else {
        return [points[start]];
      }
    };

    const simplified = rdp(contour, 0, contour.length - 1, epsilon);
    simplified.push(contour[contour.length - 1]);
    return simplified.length >= 3 ? simplified : contour;
  }

  private perpendicularDistance(point: Point, lineStart: Point, lineEnd: Point): number {
    const dx = lineEnd.x - lineStart.x;
    const dy = lineEnd.y - lineStart.y;
    const norm = Math.sqrt(dx * dx + dy * dy);
    if (norm === 0) return Math.sqrt((point.x - lineStart.x) ** 2 + (point.y - lineStart.y) ** 2);
    
    const dist = Math.abs(dy * point.x - dx * point.y + lineEnd.x * lineStart.y - lineEnd.y * lineStart.x) / norm;
    return dist;
  }

  private convexHull(points: Point[]): Point[] {
    if (points.length < 3) return points;

    let start = points[0];
    for (const point of points) {
      if (point.y < start.y || (point.y === start.y && point.x < start.x)) {
        start = point;
      }
    }

    const sorted = points.slice().sort((a, b) => {
      const angleA = Math.atan2(a.y - start.y, a.x - start.x);
      const angleB = Math.atan2(b.y - start.y, b.x - start.x);
      if (angleA !== angleB) return angleA - angleB;
      const distA = (a.x - start.x) ** 2 + (a.y - start.y) ** 2;
      const distB = (b.x - start.x) ** 2 + (b.y - start.y) ** 2;
      return distA - distB;
    });

    const hull: Point[] = [sorted[0], sorted[1]];

    for (let i = 2; i < sorted.length; i++) {
      while (hull.length > 1 && this.crossProduct(hull[hull.length - 2], hull[hull.length - 1], sorted[i]) <= 0) {
        hull.pop();
      }
      hull.push(sorted[i]);
    }

    return hull;
  }

  private crossProduct(o: Point, a: Point, b: Point): number {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  }

  private nonMaxSuppression(shapes: DetectedShape[], iouThreshold: number): DetectedShape[] {
    if (shapes.length === 0) return [];

    const sorted = shapes.slice().sort((a, b) => b.confidence - a.confidence);
    const keep: DetectedShape[] = [];

    while (sorted.length > 0) {
      const current = sorted.shift()!;
      keep.push(current);

      for (let i = sorted.length - 1; i >= 0; i--) {
        const iou = this.calculateIoU(current.boundingBox, sorted[i].boundingBox);
        if (iou > iouThreshold) {
          sorted.splice(i, 1);
        }
      }
    }

    return keep;
  }

  private calculateIoU(
    a: { x: number; y: number; width: number; height: number },
    b: { x: number; y: number; width: number; height: number }
  ): number {
    const x1 = Math.max(a.x, b.x);
    const y1 = Math.max(a.y, b.y);
    const x2 = Math.min(a.x + a.width, b.x + b.width);
    const y2 = Math.min(a.y + a.height, b.y + b.height);

    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const union = a.width * a.height + b.width * b.height - intersection;

    return union > 0 ? intersection / union : 0;
  }
}

class ShapeDetectionApp {
  private detector: ShapeDetector;
  private imageInput: HTMLInputElement;
  private resultsDiv: HTMLDivElement;
  private testImagesDiv: HTMLDivElement;
  private evaluateButton: HTMLButtonElement;
  private evaluationResultsDiv: HTMLDivElement;
  private selectionManager: SelectionManager;
  private evaluationManager: EvaluationManager;

  constructor() {
    const canvas = document.getElementById(
      "originalCanvas"
    ) as HTMLCanvasElement;
    this.detector = new ShapeDetector(canvas);

    this.imageInput = document.getElementById("imageInput") as HTMLInputElement;
    this.resultsDiv = document.getElementById("results") as HTMLDivElement;
    this.testImagesDiv = document.getElementById(
      "testImages"
    ) as HTMLDivElement;
    this.evaluateButton = document.getElementById(
      "evaluateButton"
    ) as HTMLButtonElement;
    this.evaluationResultsDiv = document.getElementById(
      "evaluationResults"
    ) as HTMLDivElement;

    this.selectionManager = new SelectionManager();
    this.evaluationManager = new EvaluationManager(
      this.detector,
      this.evaluateButton,
      this.evaluationResultsDiv
    );

    this.setupEventListeners();
    this.loadTestImages().catch(console.error);
  }

  private setupEventListeners(): void {
    this.imageInput.addEventListener("change", async (event) => {
      const file = (event.target as HTMLInputElement).files?.[0];
      if (file) {
        await this.processImage(file);
      }
    });

    this.evaluateButton.addEventListener("click", async () => {
      const selectedImages = this.selectionManager.getSelectedImages();
      await this.evaluationManager.runSelectedEvaluation(selectedImages);
    });
  }

  private async processImage(file: File): Promise<void> {
    try {
      this.resultsDiv.innerHTML = "<p>Processing...</p>";

      const imageData = await this.detector.loadImage(file);
      const results = await this.detector.detectShapes(imageData);

      this.displayResults(results);
    } catch (error) {
      this.resultsDiv.innerHTML = `<p>Error: ${error}</p>`;
    }
  }

  private displayResults(results: DetectionResult): void {
    const { shapes, processingTime } = results;

    let html = `
      <p><strong>Processing Time:</strong> ${processingTime.toFixed(2)}ms</p>
      <p><strong>Shapes Found:</strong> ${shapes.length}</p>
    `;

    if (shapes.length > 0) {
      html += "<h4>Detected Shapes:</h4><ul>";
      shapes.forEach((shape) => {
        html += `
          <li>
            <strong>${
              shape.type.charAt(0).toUpperCase() + shape.type.slice(1)
            }</strong><br>
            Confidence: ${(shape.confidence * 100).toFixed(1)}%<br>
            Center: (${shape.center.x.toFixed(1)}, ${shape.center.y.toFixed(
          1
        )})<br>
            Area: ${shape.area.toFixed(1)}px²
          </li>
        `;
      });
      html += "</ul>";
    } else {
      html +=
        "<p>No shapes detected. Please implement the detection algorithm.</p>";
    }

    this.resultsDiv.innerHTML = html;
  }

  private async loadTestImages(): Promise<void> {
    try {
      const module = await import("./test-images-data.js");
      const testImages = module.testImages;
      const imageNames = module.getAllTestImageNames();

      let html =
        '<h4>Click to upload your own image or use test images for detection. Right-click test images to select/deselect for evaluation:</h4><div class="evaluation-controls"><button id="selectAllBtn">Select All</button><button id="deselectAllBtn">Deselect All</button><span class="selection-info">0 images selected</span></div><div class="test-images-grid">';

      // Add upload functionality as first grid item
      html += `
        <div class="test-image-item upload-item" onclick="triggerFileUpload()">
          <div class="upload-icon">📁</div>
          <div class="upload-text">Upload Image</div>
          <div class="upload-subtext">Click to select file</div>
        </div>
      `;

      imageNames.forEach((imageName) => {
        const dataUrl = testImages[imageName as keyof typeof testImages];
        const displayName = imageName
          .replace(/[_-]/g, " ")
          .replace(/\.(svg|png)$/i, "");
        html += `
          <div class="test-image-item" data-image="${imageName}" 
               onclick="loadTestImage('${imageName}', '${dataUrl}')" 
               oncontextmenu="toggleImageSelection(event, '${imageName}')">
            <img src="${dataUrl}" alt="${imageName}">
            <div>${displayName}</div>
          </div>
        `;
      });

      html += "</div>";
      this.testImagesDiv.innerHTML = html;

      this.selectionManager.setupSelectionControls();

      (window as any).loadTestImage = async (name: string, dataUrl: string) => {
        try {
          const response = await fetch(dataUrl);
          const blob = await response.blob();
          const file = new File([blob], name, { type: "image/svg+xml" });

          const imageData = await this.detector.loadImage(file);
          const results = await this.detector.detectShapes(imageData);
          this.displayResults(results);

          console.log(`Loaded test image: ${name}`);
        } catch (error) {
          console.error("Error loading test image:", error);
        }
      };

      (window as any).toggleImageSelection = (
        event: MouseEvent,
        imageName: string
      ) => {
        event.preventDefault();
        this.selectionManager.toggleImageSelection(imageName);
      };

      // Add upload functionality
      (window as any).triggerFileUpload = () => {
        this.imageInput.click();
      };
    } catch (error) {
      this.testImagesDiv.innerHTML = `
        <p>Test images not available. Run 'node convert-svg-to-png.js' to generate test image data.</p>
        <p>SVG files are available in the test-images/ directory.</p>
      `;
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  new ShapeDetectionApp();
});