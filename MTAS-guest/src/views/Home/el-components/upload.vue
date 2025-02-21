<template>
    <el-upload v-model:file-list="fileList" action="http://127.0.0.1:5500/recognize" list-type="picture-card"
        :on-success="handleSuccess" :on-error="handleError" :on-exceed="handleExceed" :auto-upload="false" ref="uploadRef"
        :limit="1">
        <el-icon>
            <Plus />
        </el-icon>

        <template #file="{ file }">
            <div>
                <img class="el-upload-list__item-thumbnail" :src="file.url" alt="" />
                <span class="el-upload-list__item-actions">


                    <!-- <span class="el-upload-list__item-preview" @click="handlePictureCardPreview(file)">
                        <el-icon><zoom-in /></el-icon>
                    </span> -->

                    <!-- <span v-if="!disabled" class="el-upload-list__item-delete" @click="handleDownload(file)">
                        <el-icon>
                            <Download />
                        </el-icon>
                    </span> -->

                    <span v-if="!disabled" class="el-upload-list__item-delete" @click="handleRemove(file)">
                        <el-icon>
                            <Delete />
                        </el-icon>
                    </span>
                </span>
            </div>
        </template>

        <template #tip>
            <div class="el-upload__tip">
                jpg/png 文件必须小于 5MB
            </div>
        </template>


    </el-upload>

    <el-dialog v-model="dialogVisible">
        <img w-full :src="dialogImageUrl" alt="Preview Image" />
    </el-dialog>

    <!-- 预测按钮 -->
    <div class="card-child" style="margin-top: 25px; margin: auto;">

        <!-- <el-button class="" type="success" @click="submitUpload" style="width: 150px;">
            开始预测
        </el-button> -->

        <el-popover placement="bottom" title="提示" :width="200" trigger="click" :visible="uploadVisible"
            :content="uploadContent">
            <template #reference>

                <el-button class="" :type="loading ? 'info' : 'success'" :loading="loading" @click="submitUpload"
                    style="width: 130px; margin: 10px; ">
                    {{ loading ? '正在上传...' : '开始预测' }}
                </el-button>

            </template>
        </el-popover>


        <el-popover placement="bottom" title="提示" :width="200" trigger="click" :visible="removeVisible"
            :content="removeContent">
            <template #reference>

                <el-button class="" type="danger" @click="handleRemove" style="width: 130px; margin: 10px;">
                    移除图片
                </el-button>

            </template>
        </el-popover>


    </div>
</template>
<script lang="ts" setup>
import { Ref, ref } from 'vue'
import { Delete, Download, Plus, ZoomIn } from '@element-plus/icons-vue'

import type { UploadInstance, UploadFile, UploadUserFile } from 'element-plus'

const fileList = ref<UploadUserFile[]>([

])

const uploadVisible = ref(false)
const removeVisible = ref(false)


const removeContent = ref("")
const uploadContent = ref("")

const loading = ref(false)
let timer: ReturnType<typeof setTimeout> | null = null

// 显示提示框 （1000）
const showUploadVisible = (visible: Ref<boolean>, timer: ReturnType<typeof setTimeout> | null, time: number | 1000) => {
    if (timer) {
        clearTimeout(timer)
        timer = null
    }

    visible.value = true

    timer = setTimeout(() => {
        visible.value = false
        timer = null
    }, time)
}

// 上传事件
const uploadRef = ref<UploadInstance>()
const submitUpload = () => {

    // 逻辑判断  
    if (fileList.value.length == 0) {
        uploadContent.value = "您还没有上传图片哟！"
        showUploadVisible(uploadVisible, timer, 1000)

        return
    }

    console.log('上传！！！')
    uploadRef.value!.submit()
    loading.value = true


}

const dialogImageUrl = ref('')
const dialogVisible = ref(false)
const disabled = ref(false)




// 移除图片
const handleRemove = (file: UploadFile) => {
    console.log(file)
    const index = fileList.value.indexOf(file)
    if (fileList.value.length == 1) {
        fileList.value.splice(index, 1)
        // 调用
        removeContent.value = "移除成功(*^▽^*)"
        showUploadVisible(removeVisible, timer, 1000)
        return
    }
    // 调用
    removeContent.value = "您还没有上传图片哟！"
    showUploadVisible(removeVisible, timer, 1000)

}


// 执行超出限制时的逻辑操作
const handleExceed = (files: File[], uploadFiles: UploadUserFile[]) => {

    uploadContent.value = "上传文件数超出限制"
    showUploadVisible(uploadVisible, timer, 1000)

}


import { defineEmits } from 'vue'
// 使用defineEmits创建名称，接受一个数组
const emit = defineEmits(['clickChild'])


const handleSuccess = (response: any, uploadFile: any, uploadFiles: any) => {
    // 在这里获取上传成功后的返回数据
    // console.log('上传成功', response);

    loading.value = false   // 按钮样式改变

    uploadContent.value = "上传成功"
    showUploadVisible(uploadVisible, timer, 1000)


    //传递给父组件
    emit('clickChild', response)

    // 移除
    fileList.value = []

};



const handleError = (response: any, uploadFile: any, uploadFiles: any) => {
    // 在这里获取上传失败后的返回数据
    console.log('上传失败', response);

    loading.value = false   // 按钮样式改变

    uploadContent.value = "上传失败"
    showUploadVisible(uploadVisible, timer, 1000)

    // 移除
    fileList.value = []

};


// 预览图片
const handlePictureCardPreview = (file: UploadFile) => {
    dialogImageUrl.value = file.url!
    dialogVisible.value = true
}

// 下载图片
const handleDownload = (file: UploadFile) => {
    console.log(file)
}





</script>
