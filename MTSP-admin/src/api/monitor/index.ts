import { Result } from "@/model";
import request from "@/utils/request";
import { AxiosPromise } from "axios";
import { MonitorForm } from "./types";


/**
 * 表单提交
 * @param data 监控表单信息
 * @returns ok
 */
export function submitMonitorForm(data: MonitorForm): AxiosPromise<Result<any>> {
  return request({
    url: "/submitMonitorForm",
    method: "post",
    data,
  });
}

/**
 * 监控表单分页查询
 * @param data 监控表单分页信息
 * @returns list
 */
export function getmonitorList(page: number): AxiosPromise<Result<any>> {
  return request({
    url: `/monitorList/${page}`,
    method: "get",
  });
}

// 更新表单信息
export function updateMonitorForm(data: MonitorForm): AxiosPromise<Result<any>> {
  return request({
    url: "/updateMonitorForm",
    method: "post",
    data,
  });
}