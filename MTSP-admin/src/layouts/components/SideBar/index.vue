<template>
  <div :class="{ 'has-logo': showLogo }">
    <!-- 网站Logo -->
    <logo v-if="showLogo" :collapse="isCollapse" />
    <!-- 侧边栏 -->
    <el-scrollbar wrap-class="scrollbar-wrapper">
      <el-menu :default-active="activeMenu" :unique-opened="true" :collapse="isCollapse" :collapse-transition="false"
        :background-color="variables.menuBg" :text-color="variables.menuText"
        :active-text-color="variables.menuActiveText">
        <sidebar-item v-for="route in routes" :item="route" :key="route.path" :base-path="route.path" />
      </el-menu>
    </el-scrollbar>
  </div>
</template>

<script setup lang="ts">
import variables from '@/assets/styles/variables.module.scss';
import useStore from "@/store";
import { computed } from "vue";
import { useRoute } from 'vue-router';
import Logo from './Logo.vue';
import SidebarItem from './SidebarItem.vue';
const { app, setting, permission } = useStore();
const route = useRoute();
const isCollapse = computed(() => app.isCollapse);
const showLogo = computed(() => setting.sidebarLogo);
// const routes = computed(() => permission.routes);
const activeMenu = computed(() => route.path);

const fakeRoutes = [
  {
    path: '/index',
    meta: { title: '系统首页', icon: 'archives' },
  },
  {
    path: '',
    meta: { title: '系统管理', icon: 'system' },
    children: [
      {
        path: '/alarm',
        meta: { title: '警告信息', icon: 'dashboard' }
      },
            // {
      //   path: '/monitor',
      //   meta: { title: '控制面板', icon: 'dashboard' }
      // },
      {
        path: '/setting',
        meta: { title: '监控配置', icon: 'edit' }
      },
      {
        path: '/manager',
        meta: { title: '监控管理', icon: 'example' }
      },
      {
        path: '/user',
        meta: { title: '系统人员', icon: 'user' }
      }
    ]
  },
  // 其他路由项...
];

const routes = computed(() => fakeRoutes);
</script>

<style scoped></style>
