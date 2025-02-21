<template>
  <div class="app-container">
    <!-- 搜索栏 -->
    <el-form :model="queryParams" :inline="true">
      <el-form-item label="用户昵称">
        <el-input @keyup.enter="handleQuery" v-model="queryParams.keyword" style="width: 200px" placeholder="请输入用户昵称"
          clearable />
      </el-form-item>
      <el-form-item>
        <el-button type="primary" icon="Search" @click="handleQuery">搜索</el-button>
      </el-form-item>
    </el-form>

    <!-- 表格展示 -->
    <el-table border :data="AlarmList" v-loading="loading">
      <!-- id -->
      <el-table-column prop="id" label="用户ID" align="center" width="100"></el-table-column>
      <!-- 图片 -->
      <el-table-column prop="photo" label="照片" align="center" width="350">
        <template #default="scope">
          <img :src="scope.row.photo" class="litter-img" />
        </template>
      </el-table-column>
      <!-- 警报信息 -->
      <el-table-column prop="description" label="警报信息" align="center" width="200"></el-table-column>
      <!-- 警报地点 -->
      <el-table-column prop="location" label="警报地点" align="center" width="250" >
      </el-table-column>
      <!-- 警报时间 -->
      <el-table-column prop="create_time" width="250" label="警报时间" align="center">
        <template #default="{ row }">
          <el-date-picker :disabled="1" type="datetime" v-model="row.create_time" format="YYYY-MM-DD hh:mm:ss"
            value-format="YYYY-MM-DD hh:mm:ss">
          </el-date-picker>
        </template>
      </el-table-column>

      <!-- 监控阈值 -->
      <el-table-column prop="threshold" label="监控阈值" align="center" width="100"></el-table-column>
      <!-- 监控id -->
      <el-table-column prop="pid" label="监控id" align="center" width="100">
      </el-table-column>

      <!-- 操作 -->
      <el-table-column label="操作" align="center" width="100">
        <template #default="scope">
          <el-button type="primary" icon="Edit" link @click="openModel(scope.row)">
            编辑
          </el-button>
        </template>
      </el-table-column>
    </el-table>
    <!-- 分页 -->
    <pagination v-if="count > 0" :total="count" v-model:page="queryParams.current" v-model:limit="queryParams.size"
      @pagination="getList" />
    <!-- 添加或修改对话框 -->
    <el-dialog title="修改用户" v-model="update" width="500px" append-to-body>

      <el-form ref="userFormRef" label-width="100px" :model="userForm" :rules="rules">

        <el-form-item label="昵称" prop="username">
          <el-input placeholder="请输入昵称" v-model="userForm.username" style="width: 250px;" />
        </el-form-item>

        <el-form-item label="角色" prop="grade">
          <el-input placeholder="请输入角色" v-model="userForm.grade" style="width: 250px;" />
        </el-form-item>

        <el-form-item label="邮箱" prop="email">
          <el-input placeholder="请输入email" v-model="userForm.email" style="width: 250px;" />
        </el-form-item>

      </el-form>
      <template #footer>
        <div class="dialog-footer">
          <el-button type="primary" @click="submitForm">确 定</el-button>
          <el-button @click="update = false">取 消</el-button>
        </div>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { AlarmForm } from '@/api/alarm/types';
import { getAlarmList } from '@/api/alarm';

import { getUserList, getUserRoleList, updateUser, updateUserStatus } from '@/api/user';
import { User, UserForm, UserQuery, UserRole } from '@/api/user/types';
import { formatDate } from "@/utils/date";
import { messageConfirm, notifySuccess } from '@/utils/modal';
import { FormInstance, FormRules } from 'element-plus';
import { onMounted, reactive, ref, toRefs } from 'vue';
const userFormRef = ref<FormInstance>();
const rules = reactive<FormRules>({
  username: [{ required: true, message: "请输入昵称", trigger: "blur" }],
  roleIdList: [{ required: true, message: "角色不能为空", trigger: "click" }]
});
const data = reactive({
  count: 0,
  update: false,
  loading: false,
  queryParams: {
    current: 1,
    size: 10,
  } as UserQuery,
  typeList: [
    {
      value: 1,
      label: "邮箱",
    },
    {
      value: 2,
      label: "QQ",
    },
    {
      value: 3,
      label: "Gitee",
    },
    {
      value: 4,
      label: "Github",
    },
  ],
  userList: [] as User[],
  userForm: {} as UserForm,
  userRoleList: [] as UserRole[],
  roleIdList: [] as string[],
  AlarmList: [] as AlarmForm[],
});
const {
  count,
  update,
  loading,
  queryParams,
  typeList,
  userList,
  userForm,
  userRoleList,
  roleIdList,
  AlarmList,
} = toRefs(data);
const openModel = (user: User) => {
  roleIdList.value = [];
  userForm.value.id = user.id;
  userForm.value.username = user.username;
  userForm.value.grade = user.grade;
  userForm.value.email = user.email;
  userFormRef.value?.clearValidate();
  update.value = true;
};

const submitForm = () => {
  userFormRef.value?.validate((valid) => {
    if (valid) {
      updateUser(userForm.value).then(({ data }) => {
        if (data.flag) {
          notifySuccess(data.msg);
          getList();
        }
        update.value = false;
      })
    }
  })
};
const getList = () => {
  loading.value = true;
  getAlarmList(queryParams.value.current).then(({ data }) => {
    console.log(data)
    AlarmList.value = data.data;
    loading.value = false;
  })
};
// 搜索方法
const handleQuery = () => {
  queryParams.value.current = 1;
  // getList();
};

onMounted(() => {
  getList();
});
</script>

<style scoped>
.litter-img {
  width: 100%;
  /* 让图片宽度自适应 */
  height: 200px;
  /* 设置一个固定高度，也可以使用其他单位或百分比 */
  object-fit: cover;
  /* 让图片按比例缩放并填满容器，进行居中裁剪 */
  object-position: center;
  /* 将图片水平和垂直居中到容器 */
  overflow: hidden;
}
</style>
