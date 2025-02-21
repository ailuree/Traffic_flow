<script lang="ts">
import { defineComponent, ref } from 'vue';
import { Menu as IconMenu, Message, Setting } from '@element-plus/icons-vue'
export default defineComponent({
    data() {
        return {
            isMenuOpen: false,
        };
    },
    methods: {
        toggleMenu() {
            this.isMenuOpen = !this.isMenuOpen;
        },
    },
});
</script>
<template>
    <nav class="menu" :class="{ open: isMenuOpen }">
        <el-tooltip effect="dark" content="切换菜单样式" placement="right">
            <div class="actionsBar" @click="toggleMenu">
                <div>
                    <button id="menuBtn" type="button">
                        <el-icon><icon-menu /></el-icon>
                    </button>
                    <h3 class="menuText" :class="{ open2: isMenuOpen }">管理菜单</h3>
                </div>
            </div>
        </el-tooltip>
        <ul class="optionsBar">
            <li class="menuItem">
                <el-tooltip effect="dark" content="主页" placement="right">
                    <RouterLink to="/" class="menuOption">
                        <el-icon class="iconfont icon-shouye"></el-icon>
                        <h5 class="menuText" :class="{ open2: isMenuOpen }">主页</h5>
                    </RouterLink>
                </el-tooltip>
            </li>
            <li class="menuBreak">
                <hr>
            </li>
            <li class="menuItem">
                <el-tooltip effect="dark" content="发布" placement="right">
                    <RouterLink to="/user/uploads" class="menuOption">
                        <el-icon size="x-large">
                            <!-- 替换为 Element Plus 的图标组件 -->
                            <el-icon-plus></el-icon-plus>
                        </el-icon>
                        <h5 class="menuText" :class="{ open2: isMenuOpen }">控制面板</h5>
                    </RouterLink>
                </el-tooltip>
            </li>
        </ul>
        <div class="about" id="about"></div>
    </nav>
</template>
<!-- 
    <el-dialog v-model="dialogFormVisible" title="更新个人信息" center draggable>
        <div class="fileUpload">
            <el-upload v-model:file-list="fileList" ref="upload" action="http://localhost:8000/user/avatar/" :limit="1"
                :on-exceed="handleExceed" :auto-upload="false" :on-change="handleChange" :headers="userStore.headersObj"
                :on-success="onSuccess" :on-error="onError">
                <template #trigger>
                    <el-button class="btn" type="primary" round>选择一个文件</el-button>
                </template>
                <template #tip>
                    <div class="el-upload__tip" style="color:red;text-align: left">
                        仅限一个文件，新文件将会被覆盖
                    </div>
                </template>
            </el-upload>

        </div>
        <div class="fileUpload">
            <el-form :model="form" ref="formRef" :rules="rules" label-position="top">
                <el-form-item prop="username" label="昵称" label-width="100px" style="margin: 30px;">
                    <el-input v-model="form.username" maxlength="6" show-word-limit class="my" />
                </el-form-item>
                <el-form-item prop="signature" label="个性签名" label-width="100px" style="margin: 30px;">
                    <el-input v-model="form.signature" class="my" />
                </el-form-item>
            </el-form>
        </div>
        <template #footer>
            <span class="dialog-footer">
                <el-button @click="dialogFormVisible = false" round>取消</el-button>
                <el-button type="primary" @click="doUpdate" round>
                    确认
                </el-button>
            </span>
        </template>
    </el-dialog> -->




<style scoped>
.fileUpload {
    display: flex;
    justify-content: center;
    align-items: center;
    width: 100%;
}

.fileUpload button {
    margin-left: 20px;
    margin-bottom: 20px;
}

.overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.5);
    /* 设置透明度的背景色 */
    z-index: 9999;
    /* 设置一个较大的z-index值，确保图层位于其他内容之上 */
}

.close {
    border: 0;
    position: absolute;
    left: 70%;
    top: 18%;
    background-color: #fff;
    z-index: 1000;
    /* 设置一个较大的z-index值，确保图层位于其他内容之上 */
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

.iconfont {
    font-size: 20px;
}

.menu {
    position: absolute;
    width: 60px;
    height: auto;
    background-color: #f6f3f3;
    z-index: 2;
    top: 0px;
    bottom: 0;
    left: 10px;
    margin: auto;
    border-radius: 0.8rem;
    transition: 0.3s ease 0.15s;
    font-family: sans-serif;
    height: 100%;
}

.menu a {
    text-decoration: none;
}

.menu .actionsBar {
    width: 100%;
    height: 10%;
    padding: 0.5rem;
    overflow: hidden;
}

.menu .actionsBar div {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: space-around;
    border-radius: 0.5rem;
    transition: 0.3s ease;
}

.menu .actionsBar div button {
    background-color: transparent;
    outline: none;
    border: none;
    border-radius: 0.5rem;
    color: #000;
    width: 45px;
    height: 45px;
    transition: 0.3s ease;
    font-size: 1rem;
}

.menu .actionsBar div button:hover {
    background-color: #d5d0d0;
    color: #F19FA3;

}

.menu .actionsBar div h3 {
    width: calc(100% - 45px);
    text-align: center;
}

.menu .optionsBar {
    overflow: hidden;
    display: flex;
    width: 100%;
    height: 60%;
    padding: 0 0.5rem;
    align-items: center;
    flex-direction: column;
}

.menu .optionsBar .menuItem {
    width: 100%;
    height: 45px;
    margin: 0.3rem;
}

.menu .optionsBar .menuItem .menuOption {
    font-size: 1rem;
    outline: none;
    border: none;
    background-color: transparent;
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: space-around;
    border-radius: 0.5rem;
    transition: 0.3s ease;
}

.menu .optionsBar .menuItem .menuOption:hover {
    background-color: #d5d0d0;
    color: #f5131e;

}

.menu .optionsBar .menuItem .menuOption:hover i,
.menu .optionsBar .menuItem .menuOption:hover h5 {
    color: #f5131e;

}

.menu .optionsBar .menuItem .menuOption i {
    width: 45px;
    text-align: center;
    color: #000;
}

.menu .optionsBar .menuItem .menuOption h5 {
    width: calc(100% - 45px);
}

.menuText {
    color: #000;
    transform: translateX(-250px);
    opacity: 0;
    transition: transform 0.3s ease 0.1s;
}

.menu .about {
    width: 100%;
    height: 10%;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding: 1rem;
    flex-direction: column;
    font-family: sans-serif;
    opacity: 0;
    white-space: nowrap;
    transition: 0.3s ease 0.15s;
}

.menu.open {
    width: 240px;
    opacity: 0.9;
}

.menuText.open2 {
    opacity: 1;
    transform: translateX(0);
    text-align: center;
}

.menu .menuBreak {
    width: 100%;
    height: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.menu .menuBreak hr {
    width: 70%;
    height: 3px;
    background-color: #000;
    border: none;
    border-radius: 5px;
}

.menu .themeBar {
    overflow: hidden;
    width: 100%;
    height: 10%;
    padding: 0.5rem;
}

.menu .themeBar div {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: space-around;
    border-radius: 0.5rem;
    transition: 0.3s ease;
}

.menu .themeBar div button {
    background-color: transparent;
    outline: none;
    border: none;
    border-radius: 0.5rem;
    color: #000;
    width: 100%;
    height: 45px;
    transition: 0.3s ease;
    font-size: 1rem;
}

.menu .themeBar div button:hover {
    background-color: #d5d0d0;
    color: #f5131e;

}

.menu .menuUser {
    width: 100%;
    height: 10%;
}

.menu .menuUser a {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: space-between;
    text-decoration: none;
    padding: 0.5rem;
    position: relative;
}

.menu .menuUser a div {
    width: 45px;
    height: 45px;
    position: relative;
    border-radius: 0.5rem;
}

.menu .menuUser a div img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 0.5rem;
}

.menu .menuUser a .Username {
    width: calc(70% - 45px);
}

.menu .menuUser a p {
    width: calc(30% - 45px);
}

.menu .menuUser a:hover p {
    animation: animArrow 0.3s ease 2;
    color: #F19FA3;

}

@keyframes animArrow {
    0% {
        transform: translateX(0);
    }

    50% {
        transform: translateX(5px);
    }

    100% {
        transform: translateX(0);
    }
}

.menu .menuUser .userInfo {
    position: absolute;
    width: 20rem;
    height: 18rem;
    opacity: 0;
    pointer-events: none;
    top: 34%;
    left: 1.5rem;
    transition: 0.3s ease;
    transform: scale(0);
    transform-origin: bottom left;
}

.menu .menuUser:hover .userInfo {
    pointer-events: all;
    opacity: 1;
    transform: scale(1);
    color: #F19FA3;
}

.el-button--text {
    margin-right: 15px;
}

.el-select {
    width: 300px;
}

.el-input {
    width: 300px;
}

.dialog-footer button:first-child {
    margin-right: 10px;
}
</style>