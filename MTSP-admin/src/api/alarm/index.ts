import { Result } from "@/model";
import request from "@/utils/request";
import { AxiosPromise } from "axios";
import { AlarmForm } from "./types";

/**
 * 警告表单分页查询
 * @param data 监控表单分页信息
 * @returns list
 */
export function getAlarmList(page: number): AxiosPromise<Result<any>> {
  return request({
    url: `/alarmList/${page}`,
    method: "get",
  });
}

