<template>
    <div>
        <img src="../../../assets/images/404.gif" class="image" ref="imageRef">
    </div>
</template>
  
<script lang="ts">
import { defineComponent, onMounted, ref } from 'vue';

export default defineComponent({
    
    mounted() {
        onMounted(() => {
            this.adjustImageSize();
        });
    },
    methods: {
        adjustImageSize() {
            const imageElement = this.$refs.imageRef as HTMLImageElement;

            if (imageElement.naturalHeight > imageElement.naturalWidth) {

                imageElement.style.height = '8px';
            } else {

                imageElement.style.width = '80px';
            }
        },
    },
});

</script>
  
<style scoped>
.image {
    margin-bottom: 20px;
}
</style>
  