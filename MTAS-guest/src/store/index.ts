import useUserStore from "./modules/user";

const useStore = () => ({
  user: useUserStore(),
});

export default useStore;
