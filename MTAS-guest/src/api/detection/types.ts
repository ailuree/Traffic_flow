export interface DetectionResult {
  labels: Label[];
  after_img_path: string;
  total_objects: number;
}

export interface Label {
  id: string;
  class: string;
  cf: string;
  x1: string;
  y1: string;
  x2: string;
  y2: string;
}

export interface DetectionResponse {
  code: number;
  data: DetectionResult;
  msg: string;
} 