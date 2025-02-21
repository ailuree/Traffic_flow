import Layout from "@/layouts/index.vue";
import { createRouter, createWebHistory, RouteRecordRaw } from "vue-router";

export const constantRoutes: RouteRecordRaw[] = [
  {
    path: "/redirect",
    component: Layout,
    meta: { hidden: true },
    children: [
      {
        path: "/redirect/:path(.*)",
        component: () => import("@/views/redirect/index.vue"),
      },
    ],
  },
  {
    path: "/login",
    name: "登录",
    component: () => import("@/views/login/index.vue"),
    meta: {
      hidden: true,
    },
  },
  {
    path: "/:pathMatch(.*)*",
    component: () => import("@/views/error/404.vue"),
    meta: {
      hidden: true,
    },
  },
  {
    path: "/",
    component: Layout,
    redirect: "/index",
    children: [
      {
        path: "/index",
        component: () => import("@/views/home/index.vue"),
        name: "Index",
        meta: { title: "首页", icon: "dashboard", affix: true },
      },
      {
        path: "/monitor",
        component: () => import("@/views/system/monitorPanel/index.vue"),
        name: "monitor",
        meta: { title: "监控面板", icon: "dashboard", affix: true },
      },
      {
        path: "/manager",
        component: () => import("@/views/system/streamingManager/index.vue"),
        name: "manager",
        meta: { title: "监控管理", icon: "dashboard", affix: true },
      },
      {
        path: "/setting",
        component: () => import("@/views/system/streamingSetting/index.vue"),
        name: "setting",
        meta: { title: "监控配置", icon: "dashboard", affix: true },
      },
      {
        path: "/alarm",
        component: () => import("@/views/system/alarm/index.vue"),
        name: "alarm",
        meta: { title: "警告信息", icon: "dashboard", affix: true },
      },
      {
        path: "/user",
        component: () => import("@/views/system/user/index.vue"),
        name: "user",
        meta: { title: "用户管理", icon: "dashboard", affix: true },
      },
      
    ],
  },

];

const router = createRouter({
  history: createWebHistory(),
  routes: constantRoutes,
});

export default router;
