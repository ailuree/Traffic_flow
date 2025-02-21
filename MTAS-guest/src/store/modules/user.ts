
import { User } from "../types/user";
import { login, logout } from "@/api/login";
import { LoginForm } from "@/api/login/types";
import { defineStore } from "pinia";

const useUserStore = defineStore("useUserStore", {
  state: (): User => ({
    id: "",
    username: "游客",
    email: "",
  }),

  actions: {
    getUser(): any {
      if (this.username) {
        return this.username;
      }

      return '游客';
    },
    // 登录
    LogIn(LoginForm: LoginForm) {
      return new Promise((resolve, reject) => {
        login(LoginForm)
          .then(({ data }) => {
            // console.log(data);
            // 持久化存储
            if (data.code === 200) {
              this.username = data.data
            }
            resolve(data); // 返回结果
          })
          .catch((error) => {
            reject(error);
          });
      });
    },
    // 登出
    LogOut() {
      return new Promise((resolve, reject) => {
        logout()
          .then(({ data }) => {
            if (data.code === 200) {
              // 清除用户数据
              this.$reset(); // 调用 $reset 方法
            }

            // 其他操作...
            resolve(data); // 返回结果

          })
          .catch((error) => {
            reject(error);
          });
      });
    },
  },
  getters: {},
  persist: {
    key: "user",
    storage: localStorage,  // session关闭了就没有了
  },
});

export default useUserStore;
