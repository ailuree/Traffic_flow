
/**
 * 结果返回接口
 */
export interface Result<T> {
  /**
   * 状态码
   */
  code: number;
  /**
   * 返回信息
   */
  msg: string;
  /**
   * 返回数据
   */
  data: T;
}


/**
 * 标签类型
 */
export interface Label {
  id: string;
  class: string;
  cf: string;
  x1: string;
  y1: string;
  x2: string;
  y2: string;
}