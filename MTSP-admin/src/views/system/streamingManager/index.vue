<template>
  <div class="app-container">
    <!-- 搜索栏 -->
    <el-form :model="queryParams" :inline="true">
      <el-form-item label="监控id">
        <el-input @keyup.enter="handleQuery" v-model="queryParams.keyword" style="width: 200px" placeholder="请输入监控id"
          clearable />
      </el-form-item>
      <el-form-item>
        <el-button type="primary" icon="Search" @click="handleQuery">搜索</el-button>
      </el-form-item>
    </el-form>

    <!-- 操作按钮 -->
    <el-row :gutter="10" class="mb15">
      <el-col :span="1.5">
        <el-button type="primary" plain icon="Plus" @click="openModel(undefined)">新增</el-button>
      </el-col>
      <el-col :span="1.5">
        <el-button type="danger" plain :disabled="1" icon="Delete" @click="handleDelete(undefined)">批量删除</el-button>
      </el-col>
      <right-toolbar v-model:showSearch="showSearch" @queryTable="getList"></right-toolbar>
    </el-row>


    <!-- 表格展示 -->
    <el-table border :data="MonitorList" v-loading="loading">
      <el-table-column type="selection" width="55" />
      <!-- id -->
      <el-table-column prop="id" label="监控ID" align="center" width="100"></el-table-column>

      <!-- 视频路径 -->
      <el-table-column prop="video" label="视频路径" align="center" width="250"></el-table-column>
      <!-- 拉流路径 -->
      <el-table-column prop="url" label="拉流路径" align="center" width="250"></el-table-column>
      <!-- 监控阈值 -->
      <el-table-column prop="threshold" label="监控阈值" align="center" width="100"></el-table-column>
      <!-- 配置人员 -->
      <el-table-column prop="person" label="配置人员" align="center" width="100"></el-table-column>
      <!-- 模式 -->
      <el-table-column prop="mode" label="算法模式" align="center" width="100"></el-table-column>
      <!-- 警报状态 -->
      <el-table-column prop="is_alarm" label="警报状态" align="center" width="100"></el-table-column>
      <!-- 监控地点 -->
      <el-table-column prop="location" label="监控地点" align="center" width="300"></el-table-column>
      <!-- 创建时间 -->
      <el-table-column prop="create_time" width="250" label="创建时间" align="center">
        <template #default="{ row }">
          <el-date-picker :disabled="1" type="datetime" v-model="row.create_time" format="YYYY-MM-DD hh:mm:ss"
            value-format="YYYY-MM-DD hh:mm:ss">
          </el-date-picker>
        </template>
      </el-table-column>

      <!-- 操作 -->
      <el-table-column label="操作" align="center" width="200">
        <template #default="scope">
          <el-button type="primary" icon="Edit" link @click="openModel(scope.row)">
            编辑
          </el-button>
          <el-button type="success" icon="View" link @click="go(scope.row)">
            查看
          </el-button>
        </template>
      </el-table-column>
    </el-table>


    <!-- 分页 -->
    <pagination v-if="count > 0" :total="count" v-model:page="queryParams.current" v-model:limit="queryParams.size"
      @pagination="getList" />
    <!-- 添加或修改对话框 -->
    <el-dialog title="修改监控表单" v-model="update" width="500px" append-to-body>



      <el-form ref="FormRef" label-width="100px" :model="monitorForm" :rules="rules">


        <el-form-item label="视频路径" prop="video">
          <el-input placeholder="请输入视频路径" v-model="monitorForm.video" style="width: 250px;" />
        </el-form-item>

        <el-form-item label="拉流路径" prop="video">
          <el-input placeholder="请输入拉流路径" v-model="monitorForm.url" style="width: 250px;" />
        </el-form-item>

        <el-form-item label="监控阈值" prop="threshold">
          <el-input placeholder="请输入监控阈值" v-model="monitorForm.threshold" style="width: 250px;" />
        </el-form-item>

        <el-form-item label="配置人员" prop="person">
          <el-select v-model="monitorForm.person" placeholder="请选择配置人员">
            <el-option label="Pan" value="Pan" />
            <el-option label="Rock" value="Rock" />
          </el-select>
        </el-form-item>

        <el-form-item label="模式选择">
          <el-radio-group v-model="monitorForm.mode">
            <el-radio label="快速模式" />
            <el-radio label="精确模式" />
          </el-radio-group>
        </el-form-item>

        <el-form-item label="开启警报">
          <el-switch v-model="monitorForm.is_alarm" />
        </el-form-item>

        <el-form-item label="监控地点" prop="location">
          <el-input placeholder="请输入监控地点" v-model="monitorForm.location" style="width: 250px;" />
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
import { log } from 'console';
import { MonitorForm } from "@/api/monitor/types";
import { getmonitorList, updateMonitorForm } from '@/api/monitor';

import router from "@/router";
import { formatDate } from "@/utils/date";

import { getUserList, getUserRoleList, updateUser, updateUserStatus } from '@/api/user';
import { User, UserForm, UserQuery, UserRole } from '@/api/user/types';
import { messageConfirm, notifySuccess } from '@/utils/modal';
import { FormInstance, FormRules } from 'element-plus';
import { onMounted, reactive, ref, toRefs } from 'vue';
const FormRef = ref<FormInstance>();
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
  MonitorList: [] as MonitorForm[],
  monitorForm: {} as MonitorForm,
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
  MonitorList,
  monitorForm,
} = toRefs(data);

// 跳转
const go = (monitor: MonitorForm) => {
  const params = {
    id: monitor.id,
    video: monitor.video,
    person: monitor.person,
    mode: monitor.mode,
    location: monitor.location,
    is_alarm: monitor.is_alarm,
    remark: monitor.remark,
    threshold: monitor.threshold,
    url: monitor.url,
  };

  router.push({ path: "/monitor", query: params });
};

const openModel = (monitor: MonitorForm) => {

  monitorForm.value.id = monitor.id;
  monitorForm.value.threshold = monitor.threshold;
  monitorForm.value.location = monitor.location;
  monitorForm.value.mode = monitor.mode;
  monitorForm.value.person = monitor.person;
  monitorForm.value.video = monitor.video;
  monitorForm.value.url = monitor.url;
  monitorForm.value.remark = monitor.remark;
  monitorForm.value.is_alarm = false;

  update.value = true;
};

const submitForm = () => {
  FormRef.value?.validate((valid) => {
    if (valid) {
      updateMonitorForm(monitorForm.value).then(({ data }) => {
        console.log(data)
        if (data.code == 200) {
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
  getmonitorList(queryParams.value.current).then(({ data }) => {
    console.log(data)
    MonitorList.value = data.data;
    console.log(MonitorList.value)
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

<style scoped></style>
