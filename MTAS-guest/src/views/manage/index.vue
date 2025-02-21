<template>
    <el-container>

        <!-- <el-header>
            <div>
                <h1 class="div_center">
                    <el-icon style="margin-right: 30px; margin-top: 1px">
                        <setting />
                    </el-icon>
                    MTAS管理系统
                </h1>
            </div>
        </el-header> -->

        <el-container>

            <!-- 侧边栏 -->
            <el-aside width="200px">

                <!-- 标题栏 -->
                <div>
                    <h1 class="div_center">
                        <el-icon style="margin-right: 8px; margin-top: 1px">
                            <setting />
                        </el-icon>
                        MTAS管理系统
                    </h1>
                </div>

                <!-- 选择栏 -->
                <el-scrollbar>

                    <el-menu>
                        <el-menu-item index="1"><el-icon><icon-menu /></el-icon>系统主页</el-menu-item>
                    </el-menu>

                    <el-menu>
                        <el-menu-item index="2"><el-icon><icon-menu /></el-icon>监控面板</el-menu-item>
                    </el-menu>

                    <el-menu>
                        <el-sub-menu index="3">
                            <template #title>
                                <el-icon><icon-menu /></el-icon>视频流模块
                            </template>
                            <el-menu-item-group>
                                <el-menu-item index="3-1">视频流管理</el-menu-item>
                            </el-menu-item-group>
                            <el-menu-item-group>
                                <el-menu-item index="3-2">视频流设置</el-menu-item>
                            </el-menu-item-group>
                        </el-sub-menu>
                    </el-menu>

                    <el-menu>
                        <el-sub-menu index="4">
                            <template #title>
                                <el-icon><icon-menu /></el-icon>报警模块
                            </template>
                            <el-menu-item-group>
                                <el-menu-item index="3-1">报警联络人</el-menu-item>
                            </el-menu-item-group>
                            <el-menu-item-group>
                                <el-menu-item index="3-2">报警视频</el-menu-item>
                            </el-menu-item-group>
                            <el-menu-item-group>
                                <el-menu-item index="3-2">报警级别</el-menu-item>
                            </el-menu-item-group>
                        </el-sub-menu>
                    </el-menu>

                    <el-menu>
                        <el-sub-menu index="3">
                            <template #title>
                                <el-icon><icon-menu /></el-icon>系统算法
                            </template>
                            <el-menu-item-group>
                                <el-menu-item index="3-1">算法管理</el-menu-item>
                            </el-menu-item-group>
                            <el-menu-item-group>
                                <el-menu-item index="3-2">算法设置</el-menu-item>
                            </el-menu-item-group>
                        </el-sub-menu>
                    </el-menu>
                </el-scrollbar>
            </el-aside>


            <el-main class="maxSize" id="content">
                <RouterView :key="$route.fullPath" />
            </el-main>

        </el-container>

    </el-container>
</template>
  
<style scoped>
.maxSize {
    height: 100%;
    overflow-y: scroll;
}
</style>