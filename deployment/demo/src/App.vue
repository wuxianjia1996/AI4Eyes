<script setup lang="ts">
// @ts-ignore
import MainPage from "../pages/MainPage.vue";
</script>

<template>
  <div class="background" >
    <main-page></main-page>
  </div>
</template>

<style scoped>
.background {
  background-color: #ecf3fe;
  background-size: cover;
  background-position: center;
  height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;

  margin: 0;

  place-items: center;
}
</style>
