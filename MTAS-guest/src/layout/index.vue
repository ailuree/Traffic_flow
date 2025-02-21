<template>
  <div :class="classObj" class="app-wrapper">
    <div v-if="1" class="drawer-bg" @click="" />
    <!-- 侧边栏 -->
    <SideBar class="sidebar-container"></SideBar>
    <div :class="{ hasTagsView: 1 }" class="main-container">
      <div :class="{ 'fixed-header': 1 }">
        <!-- 导航栏 -->
        <NavBar @setLayout="setLayout"></NavBar>
        <!-- 历史标签栏 -->
        <TagView v-if="1"></TagView>
      </div>
      <AppMain></AppMain>
      <!-- 设置 -->
      <Settings ref="settingRef"></Settings>
    </div>
  </div>
</template>

<script setup lang="ts">
// import Settings from "@/components/Settings/index.vue";
// import TagView from "@/components/TagView/index.vue";

import { useWindowSize } from "@vueuse/core";
import { computed, ref, watchEffect } from "vue";
import AppMain from "./components/AppMain/index.vue";
import NavBar from "./components/NavBar/index.vue";
import SideBar from "./components/SideBar/index.vue";


const { width } = useWindowSize();
const WIDTH = 992;
const settingRef = ref();

const classObj = computed(() => ({
  hideSidebar: 1,
  openSidebar: 1,
  mobile: 1,
}));





const setLayout = () => {
  settingRef.value.openSetting();
}
</script>

<style lang="scss" scoped>


.app-wrapper {
  // @include clearfix;
  position: relative;
  height: 100%;
  width: 100%;

  &.mobile.openSidebar {
    position: fixed;
    top: 0;
  }
}

.drawer-bg {
  background: #000;
  opacity: 0.3;
  width: 100%;
  top: 0;
  height: 100%;
  position: absolute;
  z-index: 999;
}

.fixed-header {
  position: fixed;
  top: 0;
  right: 0;
  z-index: 40;
  // width: calc(100% - #{$sideBarWidth});
  transition: width 0.28s;
}

.hideSidebar .fixed-header {
  width: calc(100% - 64px);
}

.sidebarHide .fixed-header {
  width: 100%;
}

.mobile .fixed-header {
  width: 100%;
}
</style>
