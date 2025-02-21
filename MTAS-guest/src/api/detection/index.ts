import request from '@/utils/request';
import type { DetectionResponse } from './types';

// 图片识别接口
export function detectImage(file: File) {
  const formData = new FormData();
  formData.append('file', file);
  
  return request<DetectionResponse>({
    url: '/recognize',
    method: 'post',
    data: formData,
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  });
}

// 摄像头识别接口
export function detectCamera(photo: string) {
  return request<DetectionResponse>({
    url: '/photo',
    method: 'post',
    data: {
      photo: photo
    }
  });
} 