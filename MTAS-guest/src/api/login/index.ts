import { Result } from "@/models/index";
import request from "@/utils/request";
import { AxiosPromise } from "axios";
import { LoginForm } from "./types";

/**
 * 用户登录
 * @param data 登录信息
 * @returns Token
 */
export function login(data: LoginForm): AxiosPromise<Result<any>> {
  return request({
    url: "/login",
    method: "post",
    data,
  });
}

/**
 * 用户退出
 */
export function logout(): AxiosPromise<Result<null>> {
  return request({
    url: "/logOut",
    method: "get",
  });
}
