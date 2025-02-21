import router from "@/router";
import useStore from "@/store";
import { isRelogin } from "@/utils/request";
import { getToken } from "@/utils/token";
import { ElMessage } from "element-plus";
import NProgress from "nprogress";

NProgress.configure({
  easing: "ease",
  speed: 500,
  showSpinner: false,
  trickleSpeed: 200,
  minimum: 0.3,
});

// 白名单路由
const whiteList = ["/login"];

router.beforeEach((to, from, next) => {
const { user} = useStore();
  NProgress.start();
  // 判断是否有用户名
  if (user.id) {
    if (to.path === "/login") {
      next({ path: "/" });
      NProgress.done();
    } else {
      next();
    }
  } else {
    // 未登录可以访问白名单页面(登录页面)
    if (whiteList.indexOf(to.path) !== -1) {
      next();
    } else {
      next(`/login?redirect=${to.path}`);
      NProgress.done();
    }
  }
});

router.afterEach(() => {
  NProgress.done();
});
