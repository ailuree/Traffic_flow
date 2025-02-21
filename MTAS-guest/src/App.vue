<template style="height=100px;">
  <router-view v-slot="{ Component, route }">
    <keep-alive>
      <component :is="Component" :key="route.path" />
    </keep-alive>
  </router-view>
</template>

<script setup lang="ts">
// import { getBlogInfo, report } from "@/api/blogInfo";

import useStore from '@/store';
const { user } = useStore();



onMounted(() => {
  console.log(
    "%c AI系统 %c",
    "background:#e9546b ; padding: 1px; border-radius: 3px 0 0 3px;  color: #fff; padding:5px 0;",
    "background:#ec8c69 ; padding: 1px; border-radius: 0 3px 3px 0;  color: #000; padding:5px 0;",
    "background:transparent"
  );
  //获取博客信息
  // getBlogInfo().then(({ data }) => {
  //   blog.setBlogInfo(data.data);
  // });

  // 上传访客信息
  // report();

})
</script>

<style scoped>
/* 最短距离——320px */
.app-wrapper {
  position: relative;
  min-height: 100vh;
  min-width: 320px;
}

.main-wrapper {
  display: flex;
  flex-direction: column;
  width: 100%;
  padding: 0 0 8rem;
}
</style>

