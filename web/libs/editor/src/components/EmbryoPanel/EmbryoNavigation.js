let navigation = {
  availableDays: [],
  day: 0,
  frameLabel: "",
  page: 1,
  setDay: null,
  setFrame: null,
};

const listeners = new Set();

export const getEmbryoNavigation = () => navigation;

export const setEmbryoNavigation = (nextNavigation) => {
  navigation = nextNavigation;
  listeners.forEach(listener => listener(navigation));
};

export const subscribeEmbryoNavigation = (listener) => {
  listeners.add(listener);
  listener(navigation);

  return () => listeners.delete(listener);
};
