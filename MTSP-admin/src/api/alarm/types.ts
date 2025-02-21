/**
 * 警告表单信息
 */
export interface AlarmForm {
  id: number;
  location: string;
  description?: string;
  threshold: number;
  photo: string;
  pid: number;
  create_time?: string;
  remark?: string;
}





