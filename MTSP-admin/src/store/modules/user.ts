import { login, logout } from "@/api/login";
import { LoginForm } from "@/api/login/types";
import { getUserInfo } from "@/api/user";
import { removeToken, setToken } from "@/utils/token";
import { defineStore } from "pinia";
import { UserState } from "../interface";

const useUserStore = defineStore("useUserStore", {
  state: (): UserState => ({
    id: null,
    avatar: "",
    username: "",
    roleList: [],
    permissionList: [],
  }),
  actions: {
    LogIn(LoginForm: LoginForm) {
      return new Promise((resolve, reject) => {
        login(LoginForm)
          .then(({ data }) => {
            console.log(data)
            if (data.code === 200) {
              this.id = data.data.id
              this.avatar = data.data.avatar
              this.username = data.data.username
              resolve(data); // 返回结果
            }
            else {
              reject(data.msg);
            }
          })
          .catch((error) => {
            reject(error);
          });
      });
    },
    GetInfo() {
      return new Promise((resolve, reject) => {
        getUserInfo()
          .then(({ data }) => {
            
            if (data.flag) {
              this.id = data.data.id;
              this.avatar = data.data.avatar;
              this.roleList = data.data.roleList;
              this.permissionList = data.data.permissionList;
            }
            resolve(data);
          })
          .catch((error) => {
            reject(error);
          });
      });
    },
    LogOut() {
      return new Promise((resolve, reject) => {
        logout()
          .then(() => {
            this.id = null;
            this.avatar = "";
            this.roleList = [];
            this.permissionList = [];
            // 清除用户数据
            this.$reset(); // 调用 $reset 方法
            resolve(null);
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
