<template>
  <header class="header-wrapper" :class="fixedClass">
    <!-- 切换按钮 -->
    <!-- <Toggle></Toggle> -->
    <!-- 菜单 -->

    <NavBar :class="{ sub: y > 0 }"></NavBar>
    <!-- 右侧按钮 -->
    <ul class="right">
      <!-- <li class="subitem" v-show="showLogin === 1">
        <a class="link" @click="login"><svg-icon icon-class="author"></svg-icon> 登录 </a>
      </li>
      <li class="subitem" v-show="showLogin === 0">
        <a class="link" @click="logout"><svg-icon icon-class="logout"></svg-icon> 退出 </a>
      </li> -->
      <li class="subitem">
        <a class="link" @click="goManager"><svg-icon icon-class="archives"></svg-icon> 管理中心 </a>
      </li>
    </ul>
  </header>
</template>

<script setup lang="ts">
import { ElMessage } from "element-plus";
import { ref } from 'vue';
import router from "@/router";
import { useScroll } from "@vueuse/core";
import useStore from "@/store";
// 用户信息
const { user } = useStore();

const showLogin = ref(0); // 初始化为 0
if (user.getUser() == "游客") {
  showLogin.value = 1; // 更新为 1
} else {
  showLogin.value = 0; // 更新为 0
}
const goManager = () => {
  window.open("http://127.0.0.1:8080");
};

// 登录跳转
const login = () => {

  router.push("/login")
};
// 退出
const logout = () => {
  user.LogOut().then((data: any) => {
    // "账号已退出
    console.log(data)
    if (data.code === 200) {
      ElMessage({
        message: "账号已退出",
        type: "success",
        duration: 1 * 1000,
        onClose: () => {
          window.location.reload();
        },
      });
      router.replace({ path: "/" });
    }

  }).catch(() => {

  });
};

const { y } = useScroll(window);

const fixedClass = ref("");
watch(y, (newValue, oldValue) => {
  if (newValue > 0) {
    if (newValue < oldValue) {
      fixedClass.value = "show up";
    } else {
      fixedClass.value = "show down";
    }
  } else {
    fixedClass.value = "";
  }
});
</script>

<style lang="scss" scoped>
.header-wrapper {
  position: fixed;
  display: flex;
  flex-wrap: nowrap;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  height: 3.125rem;
  padding: 0 1rem;
  text-shadow: 0 0.2rem 0.3rem rgb(0 0 0 / 50%);
  color: var(--header-text-color);
  transition: all 0.2s ease-in-out 0s;
  z-index: 9;
}

.show {
  background: var(--nav-bg);
  box-shadow: 0.1rem 0.1rem 0.2rem var(--grey-9-a1);
  text-shadow: 0 0 0.625rem var(--grey-9-a1);
  color: var(--text-color);
}

.up {
  transform: translateY(0);
}

.down {
  transform: translateY(-100%);
}

.right {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  height: 100%;

  .item {
    padding: 0.625rem 0.5rem;

  }
}

@media (max-width: 991px) {
  .header-wrapper {
    padding: 0;
  }

}
</style>